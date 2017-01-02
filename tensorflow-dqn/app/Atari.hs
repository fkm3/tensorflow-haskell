{-# LANGUAGE OverloadedStrings #-}

import Control.Monad ((>=>), forM_, forever, when, void)
import Data.ByteString.Internal (toForeignPtr)
import Data.Int (Int64)
import Data.Vector.Storable.ByteString (byteStringToVector)
import Data.Word (Word8)
import Lens.Family2 ((&), (.~))
import System.Directory (getCurrentDirectory)
import System.FilePath ((</>))
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as B8
import qualified Data.ByteString.Lazy as L
import qualified Data.Vector as V
import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as SMVector

import qualified TensorFlow.Examples.DQN as DQN
import qualified ALE

import RL.Environment
import RL.Environment.Transformer
import RL.Environment.Monitor



data Atari = Atari
    { atariSession      :: ALE.Session
    , atariScreenWidth  :: !Int
    , atariScreenHeight :: !Int
    , atariActions      :: [ALE.Action]
    }

atariPixels :: Atari -> IO (S.Vector Word8)
atariPixels a = byteStringToVector <$> ALE.pixelsGray (atariSession a)

instance Env Atari where
    envNumActions a = length (atariActions a)
    envObservationShape a = [atariScreenWidth a, atariScreenHeight a]
    envReset a = do
        ALE.resetGame (atariSession a)
        atariPixels a
    envStep a action = do
        reward <- ALE.act (atariSession a) (atariActions a !! action)
        obs <- atariPixels a
        done <- ALE.gameOver (atariSession a)
        pure (obs, fromIntegral reward, done)

instance EnvLives Atari where
    envLives = ALE.lives . atariSession


withAtari :: B.ByteString -> (Atari -> IO a) -> IO a
withAtari romPath f =
    ALE.withSession romPath $ \s -> do
        w <- ALE.screenWidth s
        h <- ALE.screenHeight s
        actions <- ALE.minimalActionSet s
        putStrLn $ "actions: " ++ show actions
        f $ Atari s w h actions


main :: IO ()
main = do
    cwd <- getCurrentDirectory
    let path = cwd </> "breakout.bin"
    let transformAtari =
            pure
            -- >=> (pure . NoopReset 0 30)  -- This is only meant for eval.
            >=> newEpisodicLife
            -- The current ALE version does this by default (but that
            -- will change if I update to HEAD).
            -- >=> newConsecutiveScreenMax
            >=> (pure . ResizeScreen 84 84)
            >=> newEnvMonitor
            >=> newFrameStack 4
            >=> (pure . FrameSkip 4)
    withAtari (B8.pack path) (transformAtari >=> DQN.play)
