{-# LANGUAGE FlexibleContexts #-}
module RL.Environment.Transformer where

import Control.Monad (replicateM)
import System.Random (randomRIO)
import Data.IORef
import Data.Word (Word8)
import qualified Data.Vector.Storable as S
import qualified Vision.Image as Vision
import qualified Vision.Primitive as Vision

import RL.Environment


data NoopReset e = NoopReset Int Int e

instance EnvLives e => EnvLives (NoopReset e) where
    envLives (NoopReset _ _ e) = envLives e

instance Env e => Env (NoopReset e) where
    envNumActions (NoopReset _ _ e) = envNumActions e
    envObservationShape (NoopReset _ _ e) = envObservationShape e
    envReset (NoopReset noop k e) = do
        _ <- envReset e
        n <- randomRIO (1, k)
        (obss, _, dones) <- unzip3 <$> replicateM n (envStep e noop)
        if or dones
           then envReset (NoopReset noop k e)
           else pure (last obss)
    envStep (NoopReset _ _ e) = envStep e


-- | Resize screen. Assumes observation shape is [width, height].
data ResizeScreen e = ResizeScreen Int Int e

resizeScreen :: Env e => Int -> Int -> e -> S.Vector Word8 -> S.Vector Word8
resizeScreen w h e pixels =
    let [origW, origH] = envObservationShape e
        origSize = Vision.ix2 origH origW
        original = Vision.Manifest origSize pixels
        resized = Vision.resize Vision.NearestNeighbor (Vision.ix2 h w) original
    in Vision.manifestVector resized

instance Env e => Env (ResizeScreen e) where
    envNumActions (ResizeScreen _ _ e) = envNumActions e
    envObservationShape (ResizeScreen w h _) = [w, h]
    envReset (ResizeScreen w h e) = resizeScreen w h e <$> envReset e
    envStep (ResizeScreen w h e) a = do
        (obs, reward, done) <- envStep e a
        pure (resizeScreen w h e obs, reward, done)


-- | Take the max value for each pixel of consecutive screens.
data ConsecutiveScreenMax e = ConsecutiveScreenMax (IORef (S.Vector Word8)) e

newConsecutiveScreenMax :: Env e => e -> IO (ConsecutiveScreenMax e)
newConsecutiveScreenMax e = do
    r <- newIORef (S.fromList [])
    pure (ConsecutiveScreenMax r e)

instance Env e => Env (ConsecutiveScreenMax e) where
    envNumActions (ConsecutiveScreenMax _ e) = envNumActions e
    envObservationShape (ConsecutiveScreenMax _ e) = envObservationShape e
    envReset (ConsecutiveScreenMax r e) = do
        obs <- envReset e
        writeIORef r obs
        pure obs
    envStep (ConsecutiveScreenMax r e) a = do
        (obs, reward, done) <- envStep e a
        prev <- readIORef r
        writeIORef r obs
        pure (S.zipWith max prev obs, reward, done)


-- | Make each screen the combination of the last k screens.
data FrameStack e = FrameStack Int (IORef [S.Vector Word8]) e

newFrameStack :: Env e => Int -> e -> IO (FrameStack e)
newFrameStack k e = do
    r <- newIORef []
    pure (FrameStack k r e)

instance Env e => Env (FrameStack e) where
    envNumActions (FrameStack _ _ e) = envNumActions e
    envObservationShape (FrameStack k _ e) = k : envObservationShape e
    envReset (FrameStack k r e) = do
        obs <- envReset e
        writeIORef r (replicate k obs)
        pure (mconcat (replicate k obs))
    envStep (FrameStack k r e) a = do
        (obs, reward, done) <- envStep e a
        prevStack <- readIORef r
        let stack = take k (obs:prevStack)
        writeIORef r stack
        pure (mconcat stack, reward, done)


-- | Perform each action k times.
data FrameSkip e = FrameSkip Int e

instance Env e => Env (FrameSkip e) where
    envNumActions (FrameSkip _ e) = envNumActions e
    envObservationShape (FrameSkip _ e) = envObservationShape e
    envReset (FrameSkip _ e) = envReset e
    envStep (FrameSkip k e) a = do
        let loop i prevReward = do
                (obs, reward, done) <- envStep e a
                if done || i == 0
                   then pure (obs, prevReward + reward, done)
                   else loop (i-1) (prevReward + reward)
        loop k 0


-- | Make each life a game over to the agent.
data EpisodicLife e = EpisodicLife (IORef (Bool, Int, S.Vector Word8)) e

newEpisodicLife :: (Env e, EnvLives e) => e -> IO (EpisodicLife e)
newEpisodicLife e = do
    r <- newIORef (True, 0, S.fromList [])
    pure (EpisodicLife r e)

instance (Env e, EnvLives e) => Env (EpisodicLife e) where
    envNumActions (EpisodicLife _ e) = envNumActions e
    envObservationShape (EpisodicLife _ e) = envObservationShape e
    envReset (EpisodicLife r e) = do
        (prevDone, prevLives, prevObs) <- readIORef r
        lives <- envLives e
        if prevDone || lives == prevLives
           then do
              obs <- envReset e
              lives' <- envLives e
              writeIORef r (False, lives', obs)
              pure obs
           else do
              writeIORef r (prevDone, lives, prevObs)
              pure prevObs
    envStep (EpisodicLife r e) a = do
        (obs, reward, done) <- envStep e a
        (_, prevLives, _) <- readIORef r
        lives <- envLives e
        writeIORef r (done, prevLives, obs)
        pure (obs, reward, done || lives < prevLives)
