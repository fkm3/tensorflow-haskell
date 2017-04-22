{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}

import Control.Monad (zipWithM, when, forM_)
import Control.Monad.IO.Class (liftIO)
import Data.Int (Int32, Int64)
import Data.List (genericLength)
import Data.ProtoLens (decodeMessageOrDie)
import Data.Word (Word8)
import Proto.Tensorflow.Core.Framework.Summary (Summary)
import qualified Data.Text.IO as T
import qualified Data.Vector as V

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Logging as TF
import qualified TensorFlow.Ops as TF

import TensorFlow.Examples.MNIST.InputData
import TensorFlow.Examples.MNIST.Parse

numPixels, numLabels :: Int64
numPixels = 28*28
numLabels = 10

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam (TF.Shape shape) =
    (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    width = head shape
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

reduceMean :: TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float
reduceMean xs = TF.mean xs (TF.scalar (0 :: Int32))

gradientDescent :: Float
                -> TF.Tensor v Float
                -> [TF.Tensor TF.Ref Float]
                -> TF.Build TF.ControlNode
gradientDescent alpha loss params = do
    grads <- TF.gradients loss params
    let applyGrad param grad =
            TF.assign param (param `TF.sub` (TF.scalar alpha `TF.mul` grad))
    controlNodes <- zipWithM applyGrad params grads
    TF.group controlNodes

data Model = Model {
      train :: [V.Vector Word8]  -- ^ images
            -> [Word8]           -- ^ labels
            -> TF.Session Summary
    , infer :: [V.Vector Word8]             -- ^ images
            -> TF.Session (V.Vector Word8)  -- ^ predictions
    , errorRate :: [V.Vector Word8]  -- ^ images
                -> [Word8]           -- ^ labels
                -> TF.Session Float
    }

encodeImageBatch :: [V.Vector Word8] -> TF.TensorData Float
encodeImageBatch xs =
    TF.encodeTensorData [genericLength xs, numPixels]
                        (fromIntegral <$> mconcat xs)

encodeLabelBatch :: [Word8] -> TF.TensorData Word8
encodeLabelBatch xs =
    TF.encodeTensorData [genericLength xs]
                        (fromIntegral <$> V.fromList xs)

selectBatch :: Int -> Int -> [a] -> [a]
selectBatch batchSize i xs = take batchSize $ drop (i * batchSize) (cycle xs)

createModel :: TF.Build Model
createModel = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
    -- Inputs.
    images <- TF.placeholder [batchSize, numPixels]
    labels <- TF.placeholder [batchSize]

    -- Variables.
    hiddenWeights <- TF.initializedVariable =<< randomParam [numPixels, 500]
    logitWeights <- TF.initializedVariable =<< randomParam [500, numLabels]
    let trainableVariables = [hiddenWeights, logitWeights]

    -- Network.
    let hidden = TF.relu (images `TF.matMul` hiddenWeights)
        logits = hidden `TF.matMul` logitWeights

    -- Inference.
    predict <- TF.render $ TF.cast $ TF.argMax logits (TF.scalar (1 :: Int32))

    -- Evaluation.
    let correctPredictions = TF.equal predict labels
    errorRateTensor <- TF.render $ 1 - reduceMean (TF.cast correctPredictions)

    -- Training.
    -- Convert labels from digit indices to "one hot" vectors,
    -- e.g. 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
    loss <- TF.render $ reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
    trainStep <- gradientDescent 1e-5 loss trainableVariables

    TF.scalarSummary "loss" loss
    TF.scalarSummary "errorRate" errorRateTensor
    TF.histogramSummary "hiddenWeights" hiddenWeights
    TF.histogramSummary "logitWeights" logitWeights
    summaryTensor <- TF.mergeAllSummaries

    return Model {
          train = \imageBatch labelBatch -> do
              let feeds = [ TF.feed images (encodeImageBatch imageBatch)
                          , TF.feed labels (encodeLabelBatch labelBatch)
                          ]
              ((), summaryBytes) <- TF.runWithFeeds feeds (trainStep, summaryTensor)
              return (decodeMessageOrDie (TF.unScalar summaryBytes))
        , infer = \imageBatch -> do
              let feeds = [TF.feed images (encodeImageBatch imageBatch)]
              TF.runWithFeeds feeds predict
        , errorRate = \imageBatch labelBatch -> do
              let feeds = [ TF.feed images (encodeImageBatch imageBatch)
                          , TF.feed labels (encodeLabelBatch labelBatch)
                          ]
              TF.unScalar <$> TF.runWithFeeds feeds errorRateTensor
        }

main :: IO ()
main = TF.runSession $ do
    -- Read training and test data.
    trainingImages <- liftIO (readMNISTSamples =<< trainingImageData)
    trainingLabels <- liftIO (readMNISTLabels =<< trainingLabelData)
    testImages <- liftIO (readMNISTSamples =<< testImageData)
    testLabels <- liftIO (readMNISTLabels =<< testLabelData)

    -- Create the model.
    model <- TF.build createModel

    -- forM_ ([0..500] :: [Int]) $ \i -> do
    --     let images = selectBatch 100 i trainingImages
    --         labels = selectBatch 100 i trainingLabels
    --     _ <- train model images labels

    TF.withEventWriter "/tmp/tflogs/demo1" $ \eventWriter -> do
        forM_ ([0..500] :: [Int]) $ \i -> do
            let images = selectBatch 100 i trainingImages
                labels = selectBatch 100 i trainingLabels
            summary <- train model images labels
            TF.logSummary eventWriter (fromIntegral i) summary
            when (i `mod` 10 == 0) $ do
                err <- errorRate model images labels
                liftIO $ putStrLn $ "training error " ++ show (err * 100)
    liftIO $ putStrLn ""

    -- Test.
    testErr <- errorRate model testImages testLabels
    liftIO $ putStrLn $ "test error " ++ show (testErr * 100)

    -- Show some predictions.
    testPreds <- infer model testImages
    liftIO $ forM_ ([0..] :: [Int]) $ \i -> do
        putStrLn ""
        T.putStrLn $ drawMNIST $ testImages !! i
        putStrLn $ "expected " ++ show (testLabels !! i)
        putStrLn $ "     got " ++ show (testPreds V.! i)
        _ <- getLine
        return ()
