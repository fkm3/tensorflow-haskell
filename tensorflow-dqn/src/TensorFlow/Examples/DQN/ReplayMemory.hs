module TensorFlow.Examples.DQN.ReplayMemory
    ( Experience(..)
    , Experiences(..)
    , ReplayMemory
    , new
    , addExperience
    , sampleExperiences
    ) where

import Control.Concurrent.STM
import Data.Int (Int32)
import Data.Word (Word8)
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as SMVector
import Control.Monad (replicateM, forM_)
import System.Random (randomRIO)

-- An "experience". Includes a state, the action taken in that state, and the
-- resulting reward and state.
data Experience = Experience {
      expState :: !(S.Vector Word8)
    , expAction :: !Int32
    , expReward :: !Float
    , expIsTerminal :: !Bool
    , expState' :: !(S.Vector Word8)
    }

data Experiences = Experiences {
      expStates :: !(S.Vector Word8)
    , expActions :: !(S.Vector Int32)
    , expRewards :: !(S.Vector Float)
    , expIsTerminals :: !(S.Vector Bool)
    , expStates' :: !(S.Vector Word8)
    }

data ReplayMemory = ReplayMemory {
      _size :: Int
    , _stateSize :: Int
      -- The number of experiences available.
    , _numExperiences :: TVar Int
      -- The next position to write an experience to.
    , _experienceOffset :: TVar Int
      -- Experiences.
    , _states :: !(SMVector.IOVector Word8)
    , _actions :: !(SMVector.IOVector Int32)
    , _rewards :: !(SMVector.IOVector Float)
    , _isTerminals :: !(SMVector.IOVector Bool)
    , _states' :: !(SMVector.IOVector Word8)
    }

new :: Int -> Int -> IO ReplayMemory
new memorySize stateSize =
    ReplayMemory
    <$> pure memorySize
    <*> pure stateSize
    <*> newTVarIO 0
    <*> newTVarIO 0
    <*> SMVector.new (memorySize * stateSize)
    <*> SMVector.new memorySize
    <*> SMVector.new memorySize
    <*> SMVector.new memorySize
    <*> SMVector.new (memorySize * stateSize)

addExperience :: Experience -> ReplayMemory -> IO ()
addExperience e r = do
    i <- atomically $ do
        i <- readTVar (_experienceOffset r)
        n <- readTVar (_numExperiences r)
        writeTVar (_experienceOffset r) ((i + 1) `mod` _size r)
        writeTVar (_numExperiences r)   (min (n + 1) (_size r))
        return i

    let stateSize = _stateSize r
    S.copy (SMVector.slice (i*stateSize) stateSize (_states r)) (expState e)
    SMVector.write (_actions r) i (expAction e)
    SMVector.write (_rewards r) i (expReward e)
    SMVector.write (_isTerminals r) i (expIsTerminal e)
    S.copy (SMVector.slice (i*stateSize) stateSize (_states' r)) (expState' e)

sampleExperiences :: Int -> ReplayMemory -> IO Experiences
sampleExperiences n r = do
    numExp <- readTVarIO (_numExperiences r)
    indices <- replicateM n (randomRIO (0, numExp - 1))

    let copySimple src = S.generateM n (\i -> SMVector.read src (indices !! i))
        stateSize = _stateSize r
        copyStates src = do
            dst <- SMVector.new (n * stateSize)
            forM_ (zip [0..] indices) $ \(i, j) ->
                SMVector.copy
                    (SMVector.slice (i * stateSize) stateSize dst)
                    (SMVector.slice (j * stateSize) stateSize src)
            S.unsafeFreeze dst

    Experiences <$> copyStates (_states r)
                <*> copySimple (_actions r)
                <*> copySimple (_rewards r)
                <*> copySimple (_isTerminals r)
                <*> copyStates (_states' r)
