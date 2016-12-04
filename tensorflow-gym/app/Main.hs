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

import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64, Int32)
import Control.Monad (forM_, zipWithM, unless)
import Control.Monad.Extra (whileM)
import Text.Printf (printf)
import qualified OpenAI.Gym as Gym
import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF (max, select, square)
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Gradient as TF
import System.Random (randomRIO)
import qualified Data.Vector as V
import qualified Data.Vector.Storable as S

main :: IO ()
main = do
    let envID = "CartPole-v1"
    Gym.withEnv' "http://127.0.0.1:5000" envID $ \env -> TF.runSession $ do
        let Gym.Discrete actionCount = Gym.actionSpace env
            Gym.Box shape _ _ = Gym.observationSpace env
            stateSize = product shape
        liftIO (printf "stateSize %v actionCount %v\n" stateSize actionCount)
        model <- TF.build (createModel stateSize actionCount)

        let loop state = do
                greed <- liftIO (randomRIO (0 :: Double, 1))
                action <- if greed < 0.95
                    then policy model state
                    else liftIO (randomRIO (0, actionCount - 1))
                Gym.StepResult nextState reward done <-
                    liftIO (Gym.step env (S.singleton (fromIntegral action)) False)
                loss <- train model [state] [action] [reward] [nextState] [done]
                --liftIO (print ([state], [action], [reward], [nextState], [done]))
                --liftIO (printf "action %v reward %v\n" action reward)
                --liftIO (printf "Loss: %v\n" loss)
                if done
                   then return reward
                   else (+reward) <$> loop nextState

        forM_ [1..] $ \i -> do
            state <- liftIO (Gym.reset env)
            totalReward <- loop state
            liftIO (printf "Episode %v finished. Total reward: %v\n"
                    (i :: Int) totalReward)

                

data Model = Model {
      train :: [S.Vector Float]  -- ^ states
            -> [Int64]           -- ^ actions
            -> [Float]           -- ^ rewards
            -> [S.Vector Float]  -- ^ next states
            -> [Bool]            -- ^ whether next state is terminal
            -> TF.Session Float
    , policy :: S.Vector Float  -- ^ state
             -> TF.Session Int64     -- ^ action
    }

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: TF.Shape -> TF.Build (TF.Tensor TF.Value Float)
randomParam (TF.Shape shape@(width:_)) =
    (* stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

createModel :: Int64 -> Int64 -> TF.Build Model
createModel stateSize actionCount = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
    states <- TF.placeholder (TF.Shape [batchSize, stateSize])
    l1Weights <-
        TF.initializedVariable =<< randomParam (TF.Shape [stateSize, 20])
    l1Biases <-
        TF.initializedVariable (TF.zeros (TF.Shape [20]))
    qWeights <-
        TF.initializedVariable =<< randomParam (TF.Shape [20, actionCount])

    let calcQValuess x =
            TF.relu ((x `TF.matMul` l1Weights) `TF.add` l1Biases) `TF.matMul` qWeights

    let qValuess = calcQValuess states
    bestAction <- TF.render (TF.argMax qValuess (TF.scalar (1 :: Int32)))

    let params = [l1Weights, l1Biases, qWeights]

    -- Create training op.
    actions <- TF.placeholder (TF.Shape [batchSize])
    rewards <- TF.placeholder (TF.Shape [batchSize])
    nextStates <- TF.placeholder (TF.Shape [batchSize, stateSize])
    isTerminal <- TF.placeholder (TF.Shape [batchSize])

    let nextStateQValues = calcQValuess nextStates
        nextStateQValuesMax = TF.max nextStateQValues (TF.scalar (1 :: Int32))

    let actionsOneHot = TF.oneHot actions (fromIntegral actionCount) 1 0
        qValues = TF.sum (qValuess * actionsOneHot) (TF.scalar (1 :: Int32))
        gamma = TF.scalar 0.95
        targetQValues =
            TF.select isTerminal rewards (rewards + gamma*nextStateQValuesMax)
        losses = TF.square (qValues - targetQValues)
    loss <- TF.render (TF.mean losses (TF.scalar (0 :: Int32)))
    grads <- TF.gradients loss params

    let lr = TF.scalar 0.0001
        applyGrad param grad = TF.assign param (param `TF.sub` (lr * grad))
    trainStep <- TF.group =<< zipWithM applyGrad params grads

    return Model {
          train = \statesFeed actionsFeed rewardsFeed nextStatesFeed
                   isTerminalFeed -> do
              let len = fromIntegral (length statesFeed)
              let feeds = [ TF.feed states
                            (TF.encodeTensorData
                             (TF.Shape [len, stateSize])
                             (S.convert (mconcat statesFeed)))
                          , TF.feed actions
                            (TF.encodeTensorData
                             (TF.Shape [len])
                             (V.fromList actionsFeed))
                          , TF.feed rewards
                            (TF.encodeTensorData
                             (TF.Shape [len])
                             (V.fromList rewardsFeed))
                          , TF.feed nextStates
                            (TF.encodeTensorData
                             (TF.Shape [len, stateSize])
                             (S.convert (mconcat nextStatesFeed)))
                          , TF.feed isTerminal
                            (TF.encodeTensorData
                             (TF.Shape [len])
                             (V.fromList isTerminalFeed))
                          ]
              (_, l) <- TF.runWithFeeds feeds (trainStep, loss)
              -- (_, l, qs, tqs) <- TF.runWithFeeds feeds (trainStep, loss, qValues, targetQValues)
              -- liftIO (printf "loss %f max qvalue %v target qvalue %v\n"
              --         (TF.unScalar l)
              --         (show (qs :: V.Vector Float))
              --         (show (tqs :: V.Vector Float)))
              return (TF.unScalar l)
        , policy = \stateFeed -> do
              let feeds = [ TF.feed states (TF.encodeTensorData
                                            (TF.Shape [1, stateSize])
                                            (S.convert stateFeed))
                          ]
              TF.unScalar <$> TF.runWithFeeds feeds bestAction
        }
