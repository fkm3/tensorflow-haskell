{-# LANGUAGE OverloadedStrings #-}
module TensorFlow.Examples.DQN.GameMonitor where

import Control.Exception (assert, bracket)
import Linear (V2(..), V4(..))
import Control.Monad ((>=>), forM_, forever, when, void)
import Text.Printf (printf)
import Control.Concurrent (forkOS, threadDelay)
import Control.Concurrent.STM (atomically)
import Control.Concurrent.STM.TVar (TVar, newTVarIO, readTVarIO, writeTVar)
import Foreign.C (CInt)
import Data.IORef
import Data.Word (Word8)
import qualified Data.Vector.Storable as S
import qualified SDL

import TensorFlow.Examples.DQN (Game(..))


-- | Display the current game screen in an SDL window.
data GameMonitor g = GameMonitor (TVar (S.Vector Word8)) g

instance Game g => Game (GameMonitor g) where
    gameScreenWidth (GameMonitor _ g) = gameScreenWidth g
    gameScreenHeight (GameMonitor _ g) = gameScreenHeight g
    gameNumActions (GameMonitor _ g) = gameNumActions g
    gamePixels (GameMonitor pixelsVar g) = do
        pixels <- gamePixels g
        atomically (writeTVar pixelsVar pixels)
        pure pixels
    gameAct (GameMonitor _ g) = gameAct g
    gameIsOver (GameMonitor _ g) = gameIsOver g
    gameReset (GameMonitor _ g) = gameReset g


newGameMonitor :: Game g => g -> IO (GameMonitor g)
newGameMonitor g = do
    pixelsVar <- newTVarIO (S.fromList [])
    _ <- forkOS (renderScreenLoop (fromIntegral $ gameScreenWidth g) (fromIntegral $ gameScreenHeight g) pixelsVar)
    pure (GameMonitor pixelsVar g)


renderScreenLoop :: CInt -> CInt -> TVar (S.Vector Word8) -> IO ()
renderScreenLoop width height pixelsVar = do
    let bitDepth = 8
        pixelSize = bitDepth `div` 8
        screenBytesLen = fromIntegral (width * height * pixelSize)

    SDL.initialize [SDL.InitVideo]
    window <- SDL.createWindow
        "DQN"
        SDL.defaultWindow { SDL.windowInitialSize = V2 width height }
    SDL.showWindow window
    windowSurface <- SDL.getWindowSurface window

    let grayScale = S.fromList (map (\i -> V4 i i i 255) [0..255])
        setGrayPalette s = do
            Just palette <- SDL.formatPalette =<< SDL.surfaceFormat s
            SDL.setPaletteColors palette grayScale 0

    forever $ do
        buf <- readTVarIO pixelsVar
        if S.null buf
            then SDL.surfaceFillRect windowSurface Nothing  (V4 0 0 0 0)
            else do
                when (S.length buf /= screenBytesLen) $ error $
                    printf "bad pixel array length: expected=%d got=%d"
                        screenBytesLen (S.length buf)
                -- Create a surface from the screen bytes.
                mbuf <- S.unsafeThaw buf
                let createScreenSurface = SDL.createRGBSurfaceFrom
                        mbuf (V2 width height) (width * pixelSize) SDL.Index8

                -- Blit the surface to the window.
                bracket createScreenSurface SDL.freeSurface $ \s -> do
                    setGrayPalette s
                    void (SDL.surfaceBlit s Nothing windowSurface Nothing)
        SDL.updateWindowSurface window
        threadDelay (1000000 `div` 10)

    -- SDL.destroyWindow window
    -- SDL.quit
