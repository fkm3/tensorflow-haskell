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

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TensorFlow.Minimize
    ( Minimizer
    , minimizeWith
    , gradientDescent
    , AdamConfig(..)
    , adam
    , adam'
    ) where

import Control.Monad (zipWithM)
import Data.Default (Default(..))
import Data.List (zipWith4)
import Data.Maybe (fromMaybe)
import GHC.Exts (fromList)
import Lens.Family2.State.Strict (use)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF hiding (assign, initializedVariable)
import qualified TensorFlow.Variable as TF

-- | Functions that minimize a loss w.r.t. a set of 'TF.Variable's.
--
-- 'Minimizer's are defined as a function of the gradients instead of
-- the loss so that users can apply transformations to the gradients.
type Minimizer a =
    forall m. TF.MonadBuild m =>
    [TF.Variable a] -> [TF.Tensor TF.Value a] -> m TF.ControlNode

-- | Convenience wrapper around 'TF.gradients' and an 'Minimizer'.
minimizeWith :: (TF.MonadBuild m, TF.GradientCompatible a)
             => Minimizer a
             -> TF.Tensor v a    -- ^ Loss.
             -> [TF.Variable a]  -- ^ Parameters of the loss function.
             -> m TF.ControlNode
minimizeWith minimizer loss params = do
    grads <- TF.gradients loss params
    minimizer params grads

-- | Perform one step of the gradient descent algorithm.
gradientDescent :: TF.GradientCompatible a
                => a  -- ^ Learning rate.
                -> Minimizer a
gradientDescent learningRate params grads = TF.withNameScope "gradientDescent" $ do
    let applyGrad param grad =
            TF.assignAdd param (TF.scalar (-learningRate) `TF.mul` grad)
    TF.group =<< zipWithM applyGrad params grads

-- TODO: Support more than Float in adam.

data AdamConfig = AdamConfig
    { adamLearningRate :: Float
    , adamBeta1        :: Float
    , adamBeta2        :: Float
    , adamEpsilon      :: Float
    }

instance Default AdamConfig where
  def = AdamConfig 0.001 0.9 0.999 1e-8

-- | Perform one step of the adam algorithm.
--
-- See https://arxiv.org/abs/1412.6980.
--
-- NOTE: Currently requires all 'TF.Variable's to have a non-'Nothing'
-- 'TF.initializedValue'.
adam :: Minimizer Float
adam = adam' def

adam' :: AdamConfig -> Minimizer Float
adam' config params grads = TF.withNameScope "adam" $ do
    let lr = TF.scalar (adamLearningRate config)
        beta1 = TF.scalar (adamBeta1 config)
        beta2 = TF.scalar (adamBeta2 config)
        epsilon = TF.scalar (adamEpsilon config)
    -- Create adam state variables.
    let initVal = fromMaybe (error "no initial value") . TF.initializedValue
        zerosLikeVar v = TF.zerosLike (initVal v)
    ms <- mapM (TF.initializedVariable . zerosLikeVar) params
    vs <- mapM (TF.initializedVariable . zerosLikeVar) params
    beta1Power <- TF.initializedVariable beta1
    beta2Power <- TF.initializedVariable beta2
    -- Perform adam update.
    let applyGrad param m v =
            TF.resourceApplyAdam param m v
                                 (TF.readValue beta1Power)
                                 (TF.readValue beta2Power)
                                 lr beta1 beta2 epsilon
    updateVars <- sequence $ zipWith4 applyGrad params ms vs grads
    -- Update beta variables after adam update.
    let updateBeta betaPower beta =
            TF.withControlDependencies updateVars
                (TF.assign betaPower (TF.readValue betaPower `TF.mul` beta))
    updateBeta1 <- updateBeta beta1Power beta1
    updateBeta2 <- updateBeta beta2Power beta2
    TF.group (updateBeta1:updateBeta2:updateVars)
