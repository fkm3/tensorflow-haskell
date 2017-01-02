module TensorFlow.Examples.DQN.GameTransformers where

import Data.IORef
import Data.Word (Word8)
import qualified Data.Vector.Storable as S
import qualified Vision.Image as Vision
import qualified Vision.Primitive as Vision
import Control.Monad (replicateM)

import TensorFlow.Examples.DQN (Game(..))

-- TODO: Implement EpisodicLifeEnv from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers_deprecated.py#L53.


-- | Resize screen.
data ResizeScreen g = ResizeScreen Int Int g

instance Game g => Game (ResizeScreen g) where
    gameScreenWidth (ResizeScreen w _ _) = w
    gameScreenHeight (ResizeScreen _ h _) = h
    gameNumActions (ResizeScreen _ _ g) = gameNumActions g
    gamePixels (ResizeScreen w h g) = do
        pixels <- gamePixels g
        let origSize = Vision.ix2 (gameScreenHeight g) (gameScreenWidth g)
            original = Vision.Manifest origSize pixels
            resized = Vision.resize Vision.TruncateInteger (Vision.ix2 h w) original
        pure (Vision.manifestVector resized)
    gameAct (ResizeScreen _ _ g) = gameAct g
    gameIsOver (ResizeScreen _ _ g) = gameIsOver g
    gameReset (ResizeScreen _ _ g) = gameReset g


-- | Take the max value for each pixel of consecutive screens.
data ConsecutiveScreenMax g = ConsecutiveScreenMax (IORef (S.Vector Word8)) g

newConsecutiveScreenMax :: Game g => g -> IO (ConsecutiveScreenMax g)
newConsecutiveScreenMax g = do
    r <- newIORef (S.fromList [])
    pure (ConsecutiveScreenMax r g)

instance Game g => Game (ConsecutiveScreenMax g) where
    gameScreenWidth (ConsecutiveScreenMax _ g) = gameScreenWidth g
    gameScreenHeight (ConsecutiveScreenMax _ g) = gameScreenHeight g
    gameNumActions (ConsecutiveScreenMax _ g) = gameNumActions g
    gamePixels (ConsecutiveScreenMax r g) = do
        prev <- readIORef r
        cur <- gamePixels g
        writeIORef r cur
        pure $ if S.null prev
           then cur
           else S.zipWith max prev cur
    gameAct (ConsecutiveScreenMax _ g) = gameAct g
    gameIsOver (ConsecutiveScreenMax _ g) = gameIsOver g
    gameReset (ConsecutiveScreenMax _ g) = gameReset g


-- | Make each screen the combination of the last k screens.
data FrameStack g = FrameStack (IORef [S.Vector Word8]) g

newFrameStack :: Game g => g -> IO (FrameStack g)
newFrameStack g = do
    r <- newIORef []
    pure (FrameStack r g)

instance Game g => Game (FrameStack g) where
    gameScreenWidth (FrameStack _ g) = gameScreenWidth g
    gameScreenHeight (FrameStack _ g) = gameScreenHeight g
    gameNumActions (FrameStack _ g) = gameNumActions g
    gamePixels (FrameStack r g) = do
        prevStack <- readIORef r
        cur <- gamePixels g
        let stack = take 4 (cur:prevStack)
        writeIORef r stack
        pure (mconcat stack)
    gameAct (FrameStack _ g) = gameAct g
    gameIsOver (FrameStack _ g) = gameIsOver g
    gameReset (FrameStack r g) = do
        gameReset g
        cur <- gamePixels g
        writeIORef r (replicate 4 cur)


-- | Make each screen the combination of the last k screens.
data FrameSkip g = FrameSkip (IORef [S.Vector Word8]) g

newFrameSkip :: Game g => g -> IO (FrameSkip g)
newFrameSkip g = do
    r <- newIORef []
    pure (FrameSkip r g)

instance Game g => Game (FrameSkip g) where
    gameScreenWidth (FrameSkip _ g) = gameScreenWidth g
    gameScreenHeight (FrameSkip _ g) = gameScreenHeight g
    gameNumActions (FrameSkip _ g) = gameNumActions g
    gamePixels (FrameSkip _ g) = gamePixels g
    gameAct (FrameSkip _ g) = do
        rs <- replicateM 4 (gameAct g)
        pure (sum rs)
    gameIsOver (FrameSkip _ g) = gameIsOver g
    gameReset (FrameSkip _ g) = gameReset g


class GameLives g where
    gameLives :: g -> IO Int


-- | Make each life a game over to the agent.
data EpisodicLife g = EpisodicLife (IORef Int) g

newEpisodicLife :: (Game g, GameLives g) => g -> IO (EpisodicLife g)
newEpisodicLife g = do
    r <- newIORef 0
    pure (EpisodicLife r g)

instance (Game g, GameLives g) => Game (EpisodicLife g) where
    gameScreenWidth (EpisodicLife _ g) = gameScreenWidth g
    gameScreenHeight (EpisodicLife _ g) = gameScreenHeight g
    gameNumActions (EpisodicLife _ g) = gameNumActions g
    gamePixels (EpisodicLife _ g) = gamePixels g
    gameAct (EpisodicLife _ g) = gameAct g
    gameIsOver (EpisodicLife r g) = do
        over <- gameIsOver g
        prevLives <- readIORef r
        lives <- gameLives g
        pure (over || lives < prevLives)
    gameReset (EpisodicLife r g) = do
        over <- gameIsOver g
        prevLives <- readIORef r
        lives <- gameLives g
        if over || lives == prevLives
           then do
              gameReset g
              lives' <- gameLives g
              writeIORef r lives'
           else writeIORef r lives
