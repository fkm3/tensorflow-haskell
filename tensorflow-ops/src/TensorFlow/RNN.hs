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

{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE OverloadedStrings #-}

module TensorFlow.RNN
    ( RNN(..)
    , gru
    , lstm
    , rnnCompose
    ) where

import Control.Arrow ((&&&))
import Data.Int (Int32, Int64)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Initializers as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable, zeroInitializedVariable)
import qualified TensorFlow.Variable as TF
import qualified TensorFlow.GenOps.Core as TF (tanh, sigmoid)

data RNN state = RNN
    { rnnStep :: state -> TF.Tensor TF.Build Float -> TF.Build (state, TF.Tensor TF.Build Float)
    , rnnInitState :: forall v. TF.Tensor v Float -> state
    , rnnParams :: [TF.Variable Float]
    }

type GRUState = TF.Tensor TF.Build Float

gru :: Int64 -> Int64 -> TF.Build (RNN GRUState)
gru inputSize outputSize = do
    -- TODO: Create 3 large matrices instead of 9 small ones.
    wz <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    wr <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    wh <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    uz <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    ur <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    uh <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    bz <- TF.zeroInitializedVariable (TF.Shape [outputSize])
    br <- TF.zeroInitializedVariable (TF.Shape [outputSize])
    bh <- TF.zeroInitializedVariable (TF.Shape [outputSize])

    let zeroStateLike' = zeroStateLike outputSize
        step hPrev x = do
            let z = TF.sigmoid (x `TF.matMul` TF.readValue wz + hPrev `TF.matMul` TF.readValue uz + TF.readValue bz)
                r = TF.sigmoid (x `TF.matMul` TF.readValue wr + hPrev `TF.matMul` TF.readValue ur + TF.readValue br)
            h <- TF.render $
                z * hPrev + (1 - z) * TF.tanh (x `TF.matMul` TF.readValue wh + (r * hPrev) `TF.matMul` TF.readValue uh + TF.readValue bh)
            return (TF.expr h, TF.expr h)
    return (RNN step zeroStateLike' [wz, wr, wh, uz, ur, uh, bz, br, bh])

type LSTMState = (TF.Tensor TF.Build Float, TF.Tensor TF.Build Float)

lstm :: Int64 -> Int64 -> TF.Build (RNN LSTMState)
lstm inputSize outputSize = TF.withNameScope "lstm" $ do
    wf <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    wi <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    wo <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    wc <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [inputSize, outputSize])
    uf <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    ui <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    uo <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    uc <- TF.initializedVariable =<< TF.xavierInitializer (TF.Shape [outputSize, outputSize])
    bf <- TF.zeroInitializedVariable (TF.Shape [outputSize])
    bi <- TF.zeroInitializedVariable (TF.Shape [outputSize])
    bo <- TF.zeroInitializedVariable (TF.Shape [outputSize])
    bc <- TF.zeroInitializedVariable (TF.Shape [outputSize])

    let zeroStateLike' = zeroStateLike outputSize &&& zeroStateLike outputSize
        step (cPrev, hPrev) x = TF.withNameScope "lstm" $ do
            let f = TF.sigmoid (x `TF.matMul` TF.readValue wf + hPrev `TF.matMul` TF.readValue uf + TF.readValue bf + 1)
                i = TF.sigmoid (x `TF.matMul` TF.readValue wi + hPrev `TF.matMul` TF.readValue ui + TF.readValue bi)
                o = TF.sigmoid (x `TF.matMul` TF.readValue wo + hPrev `TF.matMul` TF.readValue uo + TF.readValue bo)
            c <- TF.render $
                f * cPrev + i * TF.tanh (x `TF.matMul` TF.readValue wc + hPrev `TF.matMul` TF.readValue uc + TF.readValue bc)
            h <- TF.render $ o * TF.tanh c
            return ((TF.expr c, TF.expr h), TF.expr h)
    return (RNN step zeroStateLike' [wf, wi, wo, wc, uf, ui, uo, uc, bf, bi, bo, bc])

rnnCompose :: TF.Build (RNN s) -> TF.Build (RNN s') -> TF.Build (RNN (s, s'))
rnnCompose brnn1 brnn2 = do
    rnn1 <- brnn1
    rnn2 <- brnn2
    let step (prev1, prev2) x = do
            (next1, out1) <- rnnStep rnn1 prev1 x
            (next2, out2) <- rnnStep rnn2 prev2 out1
            return ((next1, next2), out2)
    return (RNN step
                (rnnInitState rnn1 &&& rnnInitState rnn2)
                (rnnParams rnn1 ++ rnnParams rnn2))


zeroStateLike ::
    TF.TensorType a => Int64 -> TF.Tensor v a -> TF.Tensor TF.Build Float
zeroStateLike outputSize x =
    TF.fill shape 0
  where
    shape = TF.concat 0 [ TF.slice (TF.shape x) (TF.vector [0]) (TF.vector [1])
                        , TF.vector [fromIntegral outputSize]
                        ]
