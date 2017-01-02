{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorFlow.Examples.DQN where

import Control.Category ((>>>))
import Control.Monad (when, unless, zipWithM, forM_, when, replicateM, replicateM_)
import Control.Monad.Extra (om)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.State.Class (MonadState)
import Control.Monad.State.Strict (StateT, evalStateT, gets, modify)
import Control.Monad.Trans.Class (lift)
import Data.ByteString (ByteString)
import Data.Default (def)
import Data.Int (Int32, Int64)
import Data.Time (UTCTime, getCurrentTime, diffUTCTime)
import Data.Vector.Storable.ByteString (byteStringToVector, vectorToByteString)
import Data.Word (Word8)
import Lens.Family2
import Lens.Family2.State (use, (%=), (.=))
import Lens.Family2.TH
import System.Directory (doesFileExist)
import System.Random (randomRIO)
import Text.Printf (printf)
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as B8
import qualified Data.Vector.Mutable as MVector
import qualified Data.Vector.Storable as S

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF hiding  (placeholder, expandDims, restore, assign, truncatedNormal, save, scalarSummary, histogramSummary, unpack)
import qualified TensorFlow.Minimize as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Variable as TF
import qualified TensorFlow.Logging as TF
import qualified TensorFlow.Initializers as TF
import qualified TensorFlow.Ops as TF hiding (assign, initializedVariable', zeroInitializedVariable', save, restore)
import qualified TensorFlow.Layer as Layer
import Proto.Tensorflow.Core.Framework.Summary (Summary)
import Data.ProtoLens (decodeMessageOrDie)

import RL.Environment (Env(..))
import qualified TensorFlow.Examples.DQN.ReplayMemory as ReplayMemory


gamma :: Float
gamma = 0.99

-- Play randomly `observePeriod` frames before starting to train.
observePeriod :: Int
observePeriod = 10000

epochLength :: Int
epochLength = 1000

initialEpsilon :: Float
initialEpsilon = 1.0

finalEpsilon :: Float
finalEpsilon = 0.01

epsilonAnnealingPeriod :: Int
epsilonAnnealingPeriod = 200000

targetUpdateRate :: Int
targetUpdateRate = 1000

trainFreq :: Int
trainFreq = 4


data PlayState = PlayState {
      _stepCount :: Int
    , _episodeCount :: Int
    , _totalReward :: Float
    , _prevState :: S.Vector Word8
    }

$(makeLenses ''PlayState)

-- TODO: Move to flag.
enableCheckpoints :: Bool
enableCheckpoints = False

lerp a b f = a + (b - a) * min 1 (max 0 f)
epsilon i = lerp initialEpsilon finalEpsilon (fromIntegral i / fromIntegral epsilonAnnealingPeriod)

play :: Env e => e -> IO ()
play env = TF.runSession $ TF.withEventWriter "tflogs/breakout_double_dqn_dropout_biggermemory" $ \eventWriter -> do
    -- Make this faster by using Queues. Might not need to use TFRecord. Maybe
    -- just move the example generating code to another thread and push to a
    -- queue?

    let [framesPerState, width, height] = envObservationShape env
        numActions = fromIntegral $ envNumActions env

    model <- TF.build $ createModel numActions (fromIntegral framesPerState) (fromIntegral width) (fromIntegral height)
    initialize model
    when enableCheckpoints $
        om when (liftIO $ doesFileExist "atari_checkpoint") $ do
            liftIO (putStrLn "Restoring weights from checkpoint.")
            restore model "atari_checkpoint"
    replayMemory <- liftIO $
        ReplayMemory.new 80000 (fromIntegral (framesPerState * width * height))

    let step = do
            stepCount %= (+1)
            i <- use stepCount

            greed <- liftIO (randomRIO (0, 1))
            action <- if greed < epsilon i || i <= observePeriod
                then liftIO (fromIntegral <$> randomRIO (0, numActions - 1))
                else do
                    state <- use prevState
                    lift $ infer model state

            (state', reward, isTerminal) <- liftIO (envStep env (fromIntegral action))
            totalReward %= (+reward)
            state <- use prevState
            prevState .= state'
            liftIO $ ReplayMemory.addExperience
                (ReplayMemory.Experience state action reward isTerminal state')
                replayMemory

            when (i > observePeriod && i `mod` trainFreq == 0) $ do
                trainingExperiences <- liftIO (ReplayMemory.sampleExperiences 32 replayMemory)
                summary <- lift (train model trainingExperiences)
                lift (TF.logSummary eventWriter (fromIntegral i) summary)

            when (i `mod` targetUpdateRate == 0) (lift (updateTarget model))

            when (enableCheckpoints && i `mod` epochLength == 0)
                (lift (save model "atari_checkpoint"))

            unless isTerminal step

    let initialState = PlayState 0 0 0 (S.fromList [])
    flip evalStateT initialState $ replicateM_ 2000000 $ do
        episodeCount %= (+1)
        state <- liftIO (envReset env)
        prevState .= state
        totalReward .= 0
        startStep <- use stepCount
        startTime <- liftIO getCurrentTime
        step
        endStep <- use stepCount
        endTime <- liftIO getCurrentTime

        ep <- use episodeCount
        st <- use stepCount
        tr <- use totalReward
        liftIO $ printf "episode %5d step %5d reward=%f epsilon=%.2f fps=%.2f\n"
                        ep st tr
                        (epsilon st)
                        (fromIntegral (endStep - startStep) / realToFrac (endTime `diffUTCTime` startTime) :: Double)


data Model = Model {
      initialize :: TF.Session ()
    , save :: ByteString -> TF.Session ()
    , restore :: ByteString -> TF.Session ()
    , train :: ReplayMemory.Experiences -> TF.Session Summary
    , calculateLoss :: ReplayMemory.Experiences -> TF.Session Float
    , updateTarget :: TF.Session ()
    , infer :: S.Vector Word8   -- ^ states
            -> TF.Session Int32 -- ^ actions
    }


createModel :: Int64 -> Int64 -> Int64 -> Int64 -> TF.Build Model
createModel numActions framesPerState screenWidth screenHeight = do
    let conv2d = Layer.conv2d Layer.NCHW Layer.SAME
    let qValueLayer =
            Layer.liftPure ((`TF.div` 255) . TF.cast)
            >>> conv2d (8, 8) (4, 4) 32 TF.relu
            >>> conv2d (4, 4) (2, 2) 64 TF.relu
            >>> conv2d (3, 3) (1, 1) 64 TF.relu
            >>> Layer.flatten
            >>> Layer.dense 256 TF.relu
            >>> Layer.dense numActions id

    let inputShape = [-1, framesPerState, screenWidth, screenHeight]
    (params, _, mainNetwork) <-
        TF.withNameScope "main" $ Layer.build inputShape qValueLayer
    (targetParams, _, targetNetwork) <-
        TF.withNameScope "target" $ Layer.build inputShape qValueLayer

    states <- TF.placeholder (TF.Shape inputShape)
    nextStates <- TF.placeholder (TF.Shape inputShape)

    qValuess <- mainNetwork states
    nextQValuess <- mainNetwork nextStates
    nextTargetQValuess <- targetNetwork nextStates

    predict <- TF.withNameScope "predict" $ TF.render $ TF.cast $
        TF.argMax qValuess (TF.scalar (1 :: Int32))

    actions <- TF.placeholder (TF.Shape [-1])
    rewards <- TF.placeholder (TF.Shape [-1])
    isTerminals <- TF.placeholder (TF.Shape [-1])
    let greedyAction = TF.argMax nextQValuess (TF.scalar (1 :: Int32))
        greedyActionOneHot = TF.oneHot greedyAction (fromIntegral numActions) 1 0
        targetValue = TF.sum (greedyActionOneHot `TF.mul` nextTargetQValuess)
                             (1 :: TF.Tensor TF.Build Int32)
    let targets = TF.select
            isTerminals
            rewards
            (rewards `TF.add` TF.scalar gamma * targetValue)
        qValues = TF.sum (TF.oneHot actions (fromIntegral numActions) 1 0 `TF.mul` qValuess) (1 :: TF.Tensor TF.Build Int32)

    loss <- TF.render $ TF.reduceMean (huberLoss 1 targets qValues)

    mapM_ (\(i, q) -> TF.histogramSummary (B8.pack $ "action " ++ show i ++ " q values") q)
          (zip [0 :: Int ..] (TF.unpack 1 numActions qValuess))

    gradients <- TF.gradients loss params
    let gradientNorm = TF.globalNorm gradients
    clippedGradients <- mapM TF.render (TF.clipByGlobalNorm 10 gradients)
    trainStep <- TF.adam' (def { TF.adamLearningRate = 1e-4 }) params clippedGradients

    initTargetParams <-
        zipWithM TF.assign targetParams (map TF.readValue params) >>= TF.group
    -- let update t p = TF.assign t (TF.readValue t * TF.scalar 0.9 + TF.readValue p * TF.scalar 0.1)
    -- updateTargetParams <-
    --     TF.withNameScope "assigns" $
    --     TF.withControlDependencies trainStep $
    --     zipWithM update targetParams params >>= TF.group

    TF.scalarSummary "loss" loss
    TF.scalarSummary "gradientNorm" gradientNorm
    summaryTensor <- TF.mergeAllSummaries

    let experienceFeeds experiences =
            let batchSize = fromIntegral $ S.length (ReplayMemory.expIsTerminals experiences)
            in [ TF.feed states $ TF.encodeTensorData
                     (TF.Shape [batchSize, framesPerState, screenWidth, screenHeight])
                     (ReplayMemory.expStates experiences)
               , TF.feed actions $ TF.encodeTensorData
                     (TF.Shape [batchSize])
                     (ReplayMemory.expActions experiences)
               , TF.feed rewards $ TF.encodeTensorData
                     (TF.Shape [batchSize])
                     (ReplayMemory.expRewards experiences)
               , TF.feed isTerminals $ TF.encodeTensorData
                     (TF.Shape [batchSize])
                     (ReplayMemory.expIsTerminals experiences)
               , TF.feed nextStates $ TF.encodeTensorData
                     (TF.Shape [batchSize, framesPerState, screenWidth, screenHeight])
                     (ReplayMemory.expStates' experiences)
               ]

    return Model {
          initialize =
              TF.run_ initTargetParams
        , save = \path ->
              TF.run_ =<< TF.save @TF.Session path params
        , restore = \path -> do
              TF.run_ =<< mapM (TF.restore @TF.Session path) params
              TF.run_ initTargetParams
        , train = \experiences -> do
              ((), summaryBytes) <-
                  TF.runWithFeeds (experienceFeeds experiences)
                                  (trainStep, summaryTensor)
              return (decodeMessageOrDie (TF.unScalar summaryBytes))
        , calculateLoss = \experiences ->
              TF.unScalar <$> TF.runWithFeeds (experienceFeeds experiences) loss
        , updateTarget =
              TF.run_ initTargetParams
        , infer = \x ->
              S.head <$> TF.runWithFeeds
                  [TF.feed states $ TF.encodeTensorData
                       (TF.Shape [1, framesPerState, screenWidth, screenHeight])
                       x]
                  predict
        }


huberLoss ::
    (Fractional t, TF.TensorType t, TF.OneOf '[Float, Double ] t)
    => TF.Tensor v1 t
    -> TF.Tensor v2 t
    -> TF.Tensor v3 t
    -> TF.Tensor TF.Build t
huberLoss delta y y' =
    let error = y' `TF.sub` y
        absError = TF.abs error
        quadratic = TF.minimum absError delta
        -- The following expression is the same in value as
        -- tf.maximum(abs_error - delta, 0), but importantly the
        -- gradient for the expression when abs_error == delta is 0
        -- (for tf.maximum it would be 1). This is necessary to avoid
        -- doubling the gradient, since there is already a nonzero
        -- contribution to the gradient from the quadratic term.
        linear = absError `TF.sub` quadratic
     in (TF.scalar 0.5 * TF.square quadratic) `TF.add` (delta `TF.mul` linear)
