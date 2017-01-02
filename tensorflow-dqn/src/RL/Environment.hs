module RL.Environment where

import Data.Word (Word8)
import qualified Data.Vector.Storable as S

class Env e where
    envNumActions :: e -> Int
    envObservationShape :: e -> [Int]
    envReset :: e -> IO (S.Vector Word8)
    envStep :: e -> Int -> IO (S.Vector Word8, Float, Bool)

class EnvLives e where
    envLives :: e -> IO Int
