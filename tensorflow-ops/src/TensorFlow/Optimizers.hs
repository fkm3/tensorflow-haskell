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

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module TensorFlow.Optimizers
    ( gradientDescent
    , adam
    ) where

import Control.Monad (zipWithM)
import Data.List (zipWith4)
import Data.Maybe (fromMaybe)
import GHC.Exts (fromList)
import Lens.Family2.State.Strict (use)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF hiding (assign, initializedVariable)
import qualified TensorFlow.Variable as TF


gradientDescent :: forall m v. TF.MonadBuild m
                => Float
                -> TF.Tensor v Float
                -> [TF.Variable Float]
                -> m TF.ControlNode
gradientDescent alpha loss params = do
    let applyGrad param grad =
            TF.assignAdd param (TF.scalar (-alpha) `TF.mul` grad)
    TF.group =<< zipWithM applyGrad params =<< TF.gradients loss params


adam :: forall m v. TF.MonadBuild m
     => TF.Tensor v Float
     -> [TF.Variable Float]
     -> m TF.ControlNode
adam loss params = TF.withNameScope "adam" $ do
    let lr = TF.scalar 0.001
        beta1 = TF.scalar 0.9
        beta2 = TF.scalar 0.999
        epsilon = TF.scalar 1e-8
    -- Create adam state variables.
    let initVal = fromMaybe (error "no initial value") . TF.initializedValue
        zerosLikeVar v = TF.zerosLike (initVal v)
    ms <- mapM (TF.initializedVariable . zerosLikeVar) params
    vs <- mapM (TF.initializedVariable . zerosLikeVar) params
    beta1Power <- TF.initializedVariable beta1
    beta2Power <- TF.initializedVariable beta2
    -- Perform adam update.
    grads <- TF.gradients loss params
    let applyGrad param m v =
            TF.resourceApplyAdam param m v
                                 (TF.readValue beta1Power)
                                 (TF.readValue beta2Power)
                                 lr beta1 beta2 epsilon
    applyNodes <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta =
            TF.withControlDependencies applyNodes
                (TF.assign betaPower (TF.readValue betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    TF.group (updateBeta1:updateBeta2:applyNodes)
