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

module TensorFlow.Initializers
    ( xavierInitializer
    ) where


import qualified TensorFlow.Core as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.GenOps.Core as TF


-- | Performs "Xavier" initialization for weights.
--
-- This function implements the weight initialization from:
--
-- Xavier Glorot and Yoshua Bengio (2010):
--          Understanding the difficulty of training deep feedforward neural
--          networks. International conference on artificial intelligence and
--          statistics.
--
-- This initializer is designed to keep the scale of the gradients roughly the
-- same in all layers.
xavierInitializer :: TF.Shape -> TF.Build (TF.Tensor TF.Value Float)
xavierInitializer (TF.Shape shape) = do
    x <- TF.randomUniform (TF.vector shape)
    TF.render (TF.expr x * 2*limit - limit)
  where
    (fanIn, fanOut) = case shape of
        [] -> (1, 1)
        [x] -> (x, x)
        [x, y] -> (x, y)
        -- Assume this is for a convolution.
        xs -> let (y:x:rest) = reverse xs
                  receptiveFieldSize = product rest
              in (x * receptiveFieldSize, y * receptiveFieldSize)
    scale = 1 / max 1 (fromIntegral (fanIn + fanOut) / 2)
    limit = TF.scalar (sqrt (3 * scale))
