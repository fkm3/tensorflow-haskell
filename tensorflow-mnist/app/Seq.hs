-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Control.Lens ((^.), (^..), (.~), to)
import Control.Monad ((>=>), zipWithM, when, forM_)
import Control.Monad.IO.Class (liftIO)
import Control.Monad.Loops (unfoldrM)
import Data.Aeson.Lens (values, key, _String)
import Data.Char (toLower, isAscii)
import Data.Int (Int32, Int64)
import Data.List (genericLength)
import Data.List.Split (splitOn)
import Data.Maybe (fromJust)
import Data.Monoid ((<>))
import Data.Random (StdRandom(..))
import Data.Random.Extras (sample)
import Data.Random.RVar (runRVar)
import Data.Tuple (swap)
import Debug.Trace
import MonadUtils (mapAccumLM)
import System.FilePath.Glob (glob)
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Vector.Storable as S

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (sigmoid, tanh, div, rank, squeeze', multinomial, applyAdam)
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Initializers as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable, zeroInitializedVariable)
import qualified TensorFlow.RNN as TF
import qualified TensorFlow.Variable as TF


data Model = Model {
      train :: TF.TensorData Int32  -- ^ Inputs. [batch, timestep]
            -> TF.TensorData Int32  -- ^ Labels. [batch, timestep]
            -> TF.TensorData Float  -- ^ Mask. [batch, timestep]
            -> TF.Session (Float, Float)     -- ^ Loss and error rate.
    , infer :: TF.Session [Int32]
    , errorRate :: TF.TensorData Int32  -- ^ Inputs. [batch, timestep]
                -> TF.TensorData Int32  -- ^ Labels. [batch, timestep]
                -> TF.TensorData Float  -- ^ Mask. [batch, timestep]
                -> TF.Session Float
    }


createModel :: Int64 -> Int64 -> Int64 -> TF.Build Model
createModel maxSeqLen vocabSize inferLength = do
    let pack axis = TF.pack' (TF.opAttr "axis" .~ (axis :: Int64))
        unpack axis = TF.unpack' (TF.opAttr "axis" .~ (axis :: Int64))
        squeeze xs = TF.squeeze' (TF.opAttr "squeeze_dims" .~ (xs :: [Int64]))

    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
    -- Inputs.
    inputs <- TF.placeholder' (TF.opName .~ "inputs") [batchSize, maxSeqLen]
    mask <- TF.placeholder [batchSize, maxSeqLen]

    let hiddenSize = 128
    (rnn, zeroStateLike, rnnParams) <- TF.lstm vocabSize hiddenSize
    logitWeights <- TF.initializedVariable =<< TF.xavierInitializer [hiddenSize, vocabSize]
    logitBiases <- TF.zeroInitializedVariable [vocabSize]
    let allParams = rnnParams ++ [logitWeights, logitBiases]

    let sequences :: [TF.Tensor TF.Build Float]
        sequences = unpack 1 maxSeqLen (TF.oneHot inputs (fromIntegral vocabSize) 1 0)
    (_, rnnOutputs) <- mapAccumLM rnn (zeroStateLike (head sequences)) sequences
    let logits' = map (\x -> x `TF.matMul` TF.readValue logitWeights + TF.readValue logitBiases) rnnOutputs
        logits = pack 1 logits'

    let zeroInput = TF.zeros [1, vocabSize]
    inferResults <- flip unfoldrM (zeroStateLike zeroInput, zeroInput, 0) $ \(s, x, i) ->
        if i == inferLength
            then return Nothing
            else do
                (s', rnnOutput) <- rnn s x
                let logits = rnnOutput `TF.matMul` TF.readValue logitWeights + TF.readValue logitBiases
                chosenWords <- squeeze [1] <$> TF.multinomial logits 1
                x' <- TF.render (TF.oneHot chosenWords (fromIntegral vocabSize) 1 0)
                return (Just (chosenWords, (s', TF.expr x', i + 1)))
    infer <- TF.render (TF.cast $ pack 1 inferResults)

    predict <- TF.render $ TF.cast $ TF.argMax logits (TF.scalar (2 :: Int64))

    -- Create training action.
    labels <- TF.placeholder [batchSize, maxSeqLen] :: TF.Build (TF.Tensor TF.Value Int32)
    let labelVecs = TF.oneHot labels (fromIntegral vocabSize) 1 0 :: TF.Tensor TF.Build Float
        flatLogits = (TF.reshape logits (TF.vector [-1, fromIntegral vocabSize :: Int32]))
        flatLabels = (TF.reshape labelVecs (TF.vector [-1, fromIntegral vocabSize :: Int32]))
        flatMask = (TF.reshape mask (TF.vector [-1 :: Int32]))
        losses = fst $ TF.softmaxCrossEntropyWithLogits flatLogits flatLabels
    loss <- TF.render $ TF.reduceSum (losses * flatMask) `TF.div` TF.reduceSum flatMask
    --  trainStep <- TF.gradientDescent 1e-2 loss allParams
    trainStep <- TF.minimizeWith TF.adam loss allParams

    let correctPredictions = TF.equal predict labels
    errorRateTensor <-
        TF.render $ 1 - TF.reduceSum (mask `TF.mul` TF.cast correctPredictions) `TF.div` TF.reduceSum mask

    return Model {
          train = \inputsFeed labelsFeed maskFeed -> do
              let feeds = [ TF.feed inputs inputsFeed
                          , TF.feed labels labelsFeed
                          , TF.feed mask maskFeed
                          ]
              ((), TF.Scalar l, TF.Scalar e) <-
                  TF.runWithFeeds feeds (trainStep, loss, errorRateTensor)
              return (l, e)
        , infer = S.toList <$> TF.runWithFeeds [] infer
        , errorRate = \inputsFeed labelsFeed maskFeed -> TF.unScalar <$> TF.runWithFeeds [
                TF.feed inputs inputsFeed
              , TF.feed labels labelsFeed
              , TF.feed mask maskFeed
              ] errorRateTensor
        }

data Example = Example
    { exampleInputs :: S.Vector Int32
    , exampleLabels :: S.Vector Int32
    , exampleMask   :: S.Vector Float
    }

processDataset :: [String] -> (Int64, Int64, Map.Map Int32 Char, [Example])
processDataset examples =
    (fromIntegral maxSeqLen, fromIntegral vocabSize, idToWord, map toExample examples')
  where
    -- examples' = map (words . map toLower) examples
    examples' = filter (all isAscii) $ filter ((<1000) . length) $ map (map toLower) examples
    -- examples' = concatMap (map (take 1000)
    --                        . filter (not . null)
    --                        . splitOn "\n\n"
    --                        . map toLower) examples
    maxSeqLen = maximum (map length examples')
    vocab = Set.fromList (concat examples')
    -- 0 is reserved for null.
    vocabSize = Set.size vocab + 1
    wordToID = Map.fromList (zip ('▒' : Set.toList vocab) [0..])
    idToWord = Map.fromList (map swap (Map.toList wordToID))
    toExample xs = Example
        (S.fromList (map (wordToID Map.!) xs        ++ replicate (maxSeqLen - length xs) 0))
        (S.fromList (tail (map (wordToID Map.!) xs) ++ replicate (maxSeqLen - length xs + 1) 0))
        (S.fromList (replicate (length xs) 1        ++ replicate (maxSeqLen - length xs) 0))

getJokeExamples :: String -> IO [String]
getJokeExamples path = do
    contents <- readFile path
    let examples = contents ^.. values . to (\x ->
            (x ^. key "title" . _String) <> "\n\n" <> (x ^. key "body" . _String))
    return (map Text.unpack examples)

main :: IO ()
main = TF.runSession $ do
    -- let dataset = [ "Some simple examples."
    --               , "Another example."
    --               , "Third example!"
    --               ]

    -- dataset <- filter (not . null) . lines <$> liftIO (readFile "tensorflow/src/TensorFlow/Build.hs")
    -- dataset <- liftIO (mapM readFile =<< glob "*/src/**/*.hs")

    dataset <- liftIO (getJokeExamples "/opt/datasets/joke-dataset/reddit_jokes.json")

    let (maxSeqLen, vocabSize, idToWord, examples) = processDataset dataset

    -- liftIO (print examples)

    let sampleBatch = do
            xs <- liftIO (runRVar (sample 32 examples) StdRandom)
            let seqBatch = TF.encodeTensorData [genericLength xs, maxSeqLen]
                                              (mconcat (map exampleInputs xs))
                labelBatch = TF.encodeTensorData [genericLength xs, maxSeqLen]
                                                (mconcat (map exampleLabels xs))
                maskBatch = TF.encodeTensorData [genericLength xs, maxSeqLen]
                                                (mconcat (map exampleMask xs))
            return (seqBatch, labelBatch, maskBatch)

    -- Create the model.
    liftIO $ putStrLn $ "max sequence length: " ++ show maxSeqLen
    liftIO $ putStrLn $ "vocab size: " ++ show vocabSize
    liftIO $ putStrLn "Creating model"
    model <- TF.build (createModel maxSeqLen vocabSize maxSeqLen)

    let generateText = map (fromJust . flip Map.lookup idToWord) . takeWhile (/= 0) <$> infer model
    liftIO . putStrLn =<< generateText

    -- Train.
    liftIO $ putStrLn "Starting training..."
    forM_ ([0..100000] :: [Int]) $ \i -> do
        (seqBatch, labelBatch, maskBatch) <- sampleBatch
        (loss, err) <- train model seqBatch labelBatch maskBatch
        when (i `mod` 100 == 99) $ do
            liftIO $ putStrLn $ "\nstep " ++ show i ++ " training error " ++ show err ++ " loss " ++ show loss
            liftIO . putStrLn =<< generateText

    liftIO $ putStrLn "Finished training."
    liftIO . putStrLn =<< generateText
    liftIO . putStrLn =<< generateText
    liftIO . putStrLn =<< generateText
    liftIO . putStrLn =<< generateText
