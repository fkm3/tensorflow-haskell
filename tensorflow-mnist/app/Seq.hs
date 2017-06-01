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
import Data.ProtoLens (decodeMessageOrDie)
import Data.Random (StdRandom(..))
import Data.Random.Extras (sample)
import Data.Random.RVar (runRVar)
import Data.Tuple (swap)
import Debug.Trace
import MonadUtils (mapAccumLM)
import Options.Applicative as OptParse
import Proto.Tensorflow.Core.Framework.Summary (Summary)
import System.FilePath.Glob (glob)
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.Text as Text
import qualified Data.Vector.Storable as S

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (sigmoid, tanh, div, rank, squeeze', multinomial, applyAdam, l2Loss)
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Initializers as TF
import qualified TensorFlow.Logging as TF
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable, zeroInitializedVariable)
import qualified TensorFlow.RNN as TF
import qualified TensorFlow.Variable as TF

   
-- TODO: Training and dev set error diverge quickly. Try dropout?
-- Might want to try a longer run first.



data Model = Model {
      train :: TF.TensorData Int32  -- ^ Inputs. [batch, timestep]
            -> TF.TensorData Int32  -- ^ Labels. [batch, timestep]
            -> TF.TensorData Float  -- ^ Mask. [batch, timestep]
            -> TF.Session Summary
    , eval :: TF.TensorData Int32  -- ^ Inputs. [batch, timestep]
           -> TF.TensorData Int32  -- ^ Labels. [batch, timestep]
           -> TF.TensorData Float  -- ^ Mask. [batch, timestep]
           -> TF.Session Summary
    , infer :: TF.Session [Int32]
    , errorRate :: TF.TensorData Int32  -- ^ Inputs. [batch, timestep]
                -> TF.TensorData Int32  -- ^ Labels. [batch, timestep]
                -> TF.TensorData Float  -- ^ Mask. [batch, timestep]
                -> TF.Session Float
    }


createModel :: Int64 -> Int64 -> Int64 -> TF.Build Model
createModel maxSeqLen vocabSize inferLength = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
    -- Inputs.
    inputs <- TF.placeholder' (TF.opName .~ "inputs") (TF.Shape [batchSize, maxSeqLen])
    mask <- TF.placeholder (TF.Shape [batchSize, maxSeqLen])

    let hiddenSize = 512
    (rnn, zeroStateLike, rnnParams) <- TF.lstm vocabSize hiddenSize
    logitWeights <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [hiddenSize, vocabSize])
    logitBiases <- TF.zeroInitializedVariable (TF.Shape [vocabSize])
    let allParams = rnnParams ++ [logitWeights, logitBiases]

    let sequences :: [TF.Tensor TF.Build Float]
        sequences = TF.unpack 1 maxSeqLen (TF.oneHot inputs (fromIntegral vocabSize) 1 0)
    (_, rnnOutputs) <- mapAccumLM rnn (zeroStateLike (head sequences)) sequences
    let logits' = map (\x -> x `TF.matMul` TF.readValue logitWeights + TF.readValue logitBiases) rnnOutputs
        logits = TF.pack 1 logits'

    let zeroInput = TF.zeros (TF.Shape [1, vocabSize])
    inferResults <- flip unfoldrM (zeroStateLike zeroInput, zeroInput, 0) $ \(s, x, i) ->
        if i == inferLength
            then return Nothing
            else do
                (s', rnnOutput) <- rnn s x
                let logits = rnnOutput `TF.matMul` TF.readValue logitWeights + TF.readValue logitBiases
                chosenWords <- TF.squeeze [1] <$> TF.multinomial logits 1
                x' <- TF.render (TF.oneHot chosenWords (fromIntegral vocabSize) 1 0)
                return (Just (chosenWords, (s', TF.expr x', i + 1)))
    infer <- TF.render (TF.cast $ TF.pack 1 inferResults)

    predict <- TF.render $ TF.cast $ TF.argMax logits (TF.scalar (2 :: Int64))

    -- Create training action.
    labels <- TF.placeholder (TF.Shape [batchSize, maxSeqLen]) :: TF.Build (TF.Tensor TF.Value Int32)
    let labelVecs = TF.oneHot labels (fromIntegral vocabSize) 1 0 :: TF.Tensor TF.Build Float
        flatLogits = (TF.reshape logits (TF.vector [-1, fromIntegral vocabSize :: Int32]))
        flatLabels = (TF.reshape labelVecs (TF.vector [-1, fromIntegral vocabSize :: Int32]))
        flatMask = (TF.reshape mask (TF.vector [-1 :: Int32]))
        losses = fst $ TF.softmaxCrossEntropyWithLogits flatLogits flatLabels
    loss <- TF.render $ TF.reduceSum (losses * flatMask) `TF.div` TF.reduceSum flatMask
    --  trainStep <- TF.gradientDescent 1e-2 loss allParams
    gradients <- TF.gradients loss allParams
    let gradientNorm = TF.globalNorm gradients
    clippedGradients <- mapM TF.render (TF.clipByGlobalNorm 10 gradients)
    trainStep <- TF.adam allParams clippedGradients
    -- trainStep <- TF.adam allParams gradients

    let correctPredictions = TF.equal predict labels
    errorRateTensor <-
        TF.render $ 1 - TF.reduceSum (mask `TF.mul` TF.cast correctPredictions) `TF.div` TF.reduceSum mask

    TF.scalarSummary "loss" loss
    TF.scalarSummary "errorRate" errorRateTensor
    evalSummaries <- TF.mergeAllSummaries
    -- Don't include the gradient norm in the eval summaries,
    -- otherwise eval would be as expensive as training.
    TF.scalarSummary "gradientNorm" gradientNorm
    trainSummaries <- TF.mergeAllSummaries

    return Model {
          train = \inputsFeed labelsFeed maskFeed -> do
              let feeds = [ TF.feed inputs inputsFeed
                          , TF.feed labels labelsFeed
                          , TF.feed mask maskFeed
                          ]
              decodeMessageOrDie . TF.unScalar . snd
                  <$> TF.runWithFeeds feeds (trainStep, trainSummaries)
        , eval = \inputsFeed labelsFeed maskFeed -> do
              let feeds = [ TF.feed inputs inputsFeed
                          , TF.feed labels labelsFeed
                          , TF.feed mask maskFeed
                          ]
              decodeMessageOrDie . TF.unScalar <$> TF.runWithFeeds feeds evalSummaries
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

data Options = Options { optionsRunName :: String }

optionsParser :: OptParse.Parser Options
optionsParser = Options <$>
    OptParse.strOption (OptParse.long "run_name"
                        <> OptParse.help "Run name to use for tensorboard logs.")

main :: IO ()
main = TF.runSession $ do
    opts <- liftIO (OptParse.execParser (OptParse.info optionsParser OptParse.fullDesc))

    -- let dataset = [ "Some simple examples."
    --               , "Another example."
    --               , "Third example!"
    --               ]

    -- dataset <- filter (not . null) . lines <$> liftIO (readFile "tensorflow/src/TensorFlow/Build.hs")
    -- dataset <- liftIO (mapM readFile =<< glob "*/src/**/*.hs")

    dataset <- liftIO (getJokeExamples "/opt/datasets/joke-dataset/reddit_jokes.json")

    let (maxSeqLen, vocabSize, idToWord, examples) = processDataset dataset
        (devExamples, trainExamples) = splitAt 512 examples

    let prepareBatch xs =
            let seqBatch = TF.encodeTensorData(TF.Shape  [genericLength xs, maxSeqLen])
                                              (mconcat (map exampleInputs xs))
                labelBatch = TF.encodeTensorData (TF.Shape [genericLength xs, maxSeqLen])
                                                (mconcat (map exampleLabels xs))
                maskBatch = TF.encodeTensorData (TF.Shape [genericLength xs, maxSeqLen])
                                                (mconcat (map exampleMask xs))
            in (seqBatch, labelBatch, maskBatch)
        (devSeqBatch, devLabelBatch, devMaskBatch) = prepareBatch devExamples
        sampleBatch xs = liftIO (prepareBatch <$> runRVar (sample 32 xs) StdRandom)

    -- liftIO (print examples)

    -- Create the model.
    liftIO $ putStrLn $ "max sequence length: " ++ show maxSeqLen
    liftIO $ putStrLn $ "vocab size: " ++ show vocabSize
    liftIO $ putStrLn "Creating model"
    model <- TF.build (createModel maxSeqLen vocabSize maxSeqLen)

    let generateText = map (fromJust . flip Map.lookup idToWord) . takeWhile (/= 0) <$> infer model

    -- Train.
    liftIO $ putStrLn "Starting training..."
    TF.withEventWriter ("/tmp/seq_tflogs/" <> optionsRunName opts) $ \eventWriter ->
      TF.withEventWriter ("seq_tflogs/" <> optionsRunName opts <> "_dev") $ \devEventWriter -> do
        forM_ [0..100000] $ \i -> do
            (seqBatch, labelBatch, maskBatch) <- sampleBatch trainExamples
            TF.logSummary eventWriter i =<< train model seqBatch labelBatch maskBatch
            when (i `mod` 100 == 0) $ do
                TF.logSummary devEventWriter i =<< eval model devSeqBatch devLabelBatch devMaskBatch
                liftIO . putStrLn $ "step " ++ show i
                liftIO . putStrLn =<< generateText
                liftIO . putStrLn $ "\n"

    liftIO $ putStrLn "Finished training."
    liftIO . putStrLn =<< generateText
    liftIO . putStrLn =<< generateText
    liftIO . putStrLn =<< generateText
    liftIO . putStrLn =<< generateText
