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

{-# LANGUAGE ForeignFunctionInterface #-}

module ALE
    ( Session
    , withSession
    , newSession
    , deleteSession
    , loadROM
    , resetGame
    , gameOver
    , lives
    , Action
    , act
    , legalActionSet
    , minimalActionSet
    , frameNumber
    , screenWidth
    , screenHeight
    , pixelsRGB
    , pixelsGray
    ) where

#include "ale_c_wrapper.h"

import Control.Exception (bracket, mask_)
import Data.ByteString (ByteString, useAsCString)
import Data.ByteString.Unsafe (unsafePackMallocCStringLen)
import Data.Map (Map)
import Data.Maybe (fromJust)
import Data.Tuple (swap)
import Foreign (castPtr)
import Foreign.C (CInt(..))
import Foreign.Marshal (allocaArray, mallocBytes, peekArray)
import qualified Data.Map as Map

data Action = PLAYER_A_NOOP
            | PLAYER_A_FIRE
            | PLAYER_A_UP
            | PLAYER_A_RIGHT
            | PLAYER_A_LEFT
            | PLAYER_A_DOWN
            | PLAYER_A_UPRIGHT
            | PLAYER_A_UPLEFT
            | PLAYER_A_DOWNRIGHT
            | PLAYER_A_DOWNLEFT
            | PLAYER_A_UPFIRE
            | PLAYER_A_RIGHTFIRE
            | PLAYER_A_LEFTFIRE
            | PLAYER_A_DOWNFIRE
            | PLAYER_A_UPRIGHTFIRE
            | PLAYER_A_UPLEFTFIRE
            | PLAYER_A_DOWNRIGHTFIRE
            | PLAYER_A_DOWNLEFTFIRE
            | PLAYER_B_NOOP
            | PLAYER_B_FIRE
            | PLAYER_B_UP
            | PLAYER_B_RIGHT
            | PLAYER_B_LEFT
            | PLAYER_B_DOWN
            | PLAYER_B_UPRIGHT
            | PLAYER_B_UPLEFT
            | PLAYER_B_DOWNRIGHT
            | PLAYER_B_DOWNLEFT
            | PLAYER_B_UPFIRE
            | PLAYER_B_RIGHTFIRE
            | PLAYER_B_LEFTFIRE
            | PLAYER_B_DOWNFIRE
            | PLAYER_B_UPRIGHTFIRE
            | PLAYER_B_UPLEFTFIRE
            | PLAYER_B_DOWNRIGHTFIRE
            | PLAYER_B_DOWNLEFTFIRE
            | RESET
            | UNDEFINED
            | RANDOM
            | SAVE_STATE
            | LOAD_STATE
            | SYSTEM_RESET
            | SELECT
            | LAST_ACTION_INDEX
            deriving (Show, Ord, Eq)

instance Enum Action where
    toEnum x = fromJust $ Map.lookup x actionToEnumMap
    fromEnum x = fromJust $ Map.lookup x actionFromEnumMap

actionFromEnumMap :: Map Action Int
actionFromEnumMap = Map.fromList
    [ (PLAYER_A_NOOP           , 0)
    , (PLAYER_A_FIRE           , 1)
    , (PLAYER_A_UP             , 2)
    , (PLAYER_A_RIGHT          , 3)
    , (PLAYER_A_LEFT           , 4)
    , (PLAYER_A_DOWN           , 5)
    , (PLAYER_A_UPRIGHT        , 6)
    , (PLAYER_A_UPLEFT         , 7)
    , (PLAYER_A_DOWNRIGHT      , 8)
    , (PLAYER_A_DOWNLEFT       , 9)
    , (PLAYER_A_UPFIRE         , 10)
    , (PLAYER_A_RIGHTFIRE      , 11)
    , (PLAYER_A_LEFTFIRE       , 12)
    , (PLAYER_A_DOWNFIRE       , 13)
    , (PLAYER_A_UPRIGHTFIRE    , 14)
    , (PLAYER_A_UPLEFTFIRE     , 15)
    , (PLAYER_A_DOWNRIGHTFIRE  , 16)
    , (PLAYER_A_DOWNLEFTFIRE   , 17)
    , (PLAYER_B_NOOP           , 18)
    , (PLAYER_B_FIRE           , 19)
    , (PLAYER_B_UP             , 20)
    , (PLAYER_B_RIGHT          , 21)
    , (PLAYER_B_LEFT           , 22)
    , (PLAYER_B_DOWN           , 23)
    , (PLAYER_B_UPRIGHT        , 24)
    , (PLAYER_B_UPLEFT         , 25)
    , (PLAYER_B_DOWNRIGHT      , 26)
    , (PLAYER_B_DOWNLEFT       , 27)
    , (PLAYER_B_UPFIRE         , 28)
    , (PLAYER_B_RIGHTFIRE      , 29)
    , (PLAYER_B_LEFTFIRE       , 30)
    , (PLAYER_B_DOWNFIRE       , 31)
    , (PLAYER_B_UPRIGHTFIRE    , 32)
    , (PLAYER_B_UPLEFTFIRE     , 33)
    , (PLAYER_B_DOWNRIGHTFIRE  , 34)
    , (PLAYER_B_DOWNLEFTFIRE   , 35)
    , (RESET                   , 40)
    , (UNDEFINED               , 41)
    , (RANDOM                  , 42)
    , (SAVE_STATE              , 43)
    , (LOAD_STATE              , 44)
    , (SYSTEM_RESET            , 45)
    , (SELECT                  , 46)
    , (LAST_ACTION_INDEX       , 50)
    ]

actionToEnumMap :: Map Int Action
actionToEnumMap = Map.fromList $ map swap $ Map.toList actionFromEnumMap


toCInt :: Int -> CInt
toCInt = fromIntegral

fromCInt :: CInt -> Int
fromCInt = fromIntegral


{# pointer *ALEInterface as Session newtype #}

withSession :: ByteString -> (Session -> IO a) -> IO a
withSession romPath f =
    bracket newSession deleteSession $ \sess -> do
        loadROM sess romPath
        f sess

newSession :: IO Session
newSession = {# call ALE_new as ^ #}

deleteSession :: Session -> IO ()
deleteSession = {# call ALE_del as ^ #}

loadROM :: Session -> ByteString -> IO ()
loadROM s romPath = useAsCString romPath ({# call loadROM as loadROM_ #} s)

resetGame :: Session -> IO ()
resetGame = {# call reset_game as resetGame_ #}

gameOver :: Session -> IO Bool
gameOver = fmap ((/= 0) . fromCInt) . {# call game_over as gameOver_ #}

lives :: Session -> IO Int
lives = fmap fromCInt .  {# call lives as lives_ #}

act :: Session -> Action -> IO Int
act s a = fromCInt <$> {# call act as act_ #} s (toCInt (fromEnum a))

legalActionSet :: Session -> IO [Action]
legalActionSet s = do
    numActions <- fromCInt <$> {# call getLegalActionSize as ^ #} s
    allocaArray numActions $ \actionsOut -> do
        {# call getLegalActionSet as ^ #} s actionsOut
        fmap (toEnum . fromCInt) <$> peekArray numActions actionsOut

minimalActionSet :: Session -> IO [Action]
minimalActionSet s = do
    numActions <- fromCInt <$> {# call getMinimalActionSize as ^ #} s
    allocaArray numActions $ \actionsOut -> do
        {# call getMinimalActionSet as ^ #} s actionsOut
        fmap (toEnum . fromCInt) <$> peekArray numActions actionsOut

frameNumber :: Session -> IO Int
frameNumber = fmap fromCInt . {# call getFrameNumber as ^ #}

screenWidth :: Session -> IO Int
screenWidth = fmap fromCInt .  {# call getScreenWidth as ^ #}

screenHeight :: Session -> IO Int
screenHeight = fmap fromCInt .  {# call getScreenHeight as ^ #}

pixelsRGB :: Session -> IO ByteString
pixelsRGB s = do
    w <- screenWidth s
    h <- screenHeight s
    let size = w * h * 3
    mask_ $ do
        buf <- mallocBytes size
        {# call getScreenRGB as ^ #} s buf
        unsafePackMallocCStringLen (castPtr buf, size)

pixelsGray :: Session -> IO ByteString
pixelsGray s = do
    w <- screenWidth s
    h <- screenHeight s
    let size = w * h 
    mask_ $ do
        buf <- mallocBytes size
        {# call getScreenGrayscale as ^ #} s buf
        unsafePackMallocCStringLen (castPtr buf, size)
