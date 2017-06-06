{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
module TensorFlow.Layer
    ( Layer(..)
    , build
      -- Re-export id since most users won't have imported the Category version.
    , id
    , lift
    , liftPure
    , reshape
    , dropout
    , dense
    , Conv2DFormat(..)
    , Conv2DPadding(..)
    , conv2d
    , flatten
    ) where

import Control.Monad ((<=<))
import Control.Category (Category(..))
import Data.ByteString (ByteString)
import Data.Int (Int32, Int64)
import Lens.Family2 ((.~))
import Prelude hiding ((.), id)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Initializers as TF
import qualified TensorFlow.Ops as TF hiding (initializedVariable, zeroInitializedVariable)
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Variable as TF

data Layer a b = Layer ([Int64] -> TF.Build ([TF.Variable Float], [Int64], a -> TF.Build b))

instance Category Layer where
    (Layer l2) . (Layer l1) = Layer $ \inputShape -> do
        (vs1, outputShape1, f1) <- l1 inputShape
        (vs2, outputShape2, f2) <- l2 outputShape1
        pure (vs2 ++ vs1, outputShape2, f2 <=< f1)
    id = Layer (\shape -> pure ([], shape, pure . id))


build ::
       [Int64]
    -> Layer (TF.Tensor v1 a) (TF.Tensor v2 b)
    -> TF.Build ([TF.Variable Float], [Int64], TF.Tensor v1 a -> TF.Build (TF.Tensor v2 b))
build inputShape (Layer f) = f inputShape


lift ::
       (TF.Tensor v1 a -> TF.Build (TF.Tensor v2 b))
    -> Layer (TF.Tensor v1 a) (TF.Tensor v2 b)
lift f = Layer (\shape -> pure ([], shape, f))

liftPure ::
       (TF.Tensor v1 a -> TF.Tensor v2 b)
    -> Layer (TF.Tensor v1 a) (TF.Tensor TF.Value b)
liftPure f = Layer (\shape -> pure ([], shape, TF.renderValue . f))


reshape ::
    TF.TensorType a => [Int64] -> Layer (TF.Tensor v a) (TF.Tensor TF.Value a)
reshape shape = Layer $ \_ ->
    let shape32 = map fromIntegral shape :: [Int32]
    in pure ([], shape, TF.render . flip TF.reshape (TF.vector shape32))

    
flatten :: TF.TensorType a => Layer (TF.Tensor v a) (TF.Tensor TF.Value a)
flatten = Layer $ \(batch:rest) -> do
    let outputShape = [batch, product rest]
        outputShape32 = fmap (fromIntegral :: Int64 -> Int32) outputShape
    let run x = TF.reshape x (TF.vector outputShape32)
    pure ([], outputShape, TF.render . run)


dropout ::
    TF.OneOf '[ Float, Double ] a
    => TF.Tensor v1 a -> Layer (TF.Tensor v2 a) (TF.Tensor TF.Value a)
dropout = lift . TF.dropout


dense ::
       Int64
    -> (TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float)
    -> Layer (TF.Tensor v2 Float) (TF.Tensor TF.Value Float)
dense outputSize activation = Layer $ \[batch, inputSize] -> do
    let shape = TF.Shape [inputSize, outputSize]
    filter <- TF.initializedVariable =<< TF.xavierInitializer shape
    bias <- TF.zeroInitializedVariable (TF.Shape [outputSize])
    let run x = activation (x `TF.matMul` TF.readValue filter) `TF.add` TF.readValue bias
    pure ([filter, bias], [batch, outputSize], TF.render . run)


data Conv2DFormat = NCHW | NHWC
data Conv2DPadding = VALID | SAME

conv2d ::
       Conv2DFormat
    -> Conv2DPadding
    -> (Int64, Int64)
    -> (Int64, Int64)
    -> Int64
    -> (TF.Tensor TF.Build Float -> TF.Tensor TF.Build Float)
    -> Layer (TF.Tensor v2 Float) (TF.Tensor TF.Value Float)
conv2d format padding (hFilterSize, wFilterSize) (hStride, wStride) outputSize activation =
    Layer $ \[n, c, h, w] -> do
        let shape = TF.Shape [hFilterSize, wFilterSize, c, outputSize]
        filter <- TF.initializedVariable =<< TF.xavierInitializer shape
        bias <- TF.zeroInitializedVariable (TF.Shape [outputSize, 1, 1])
        let formatString = case format of
                NCHW -> "NCHW" :: ByteString
                NHWC -> "NHWC"
            paddingString = case padding of
                VALID -> "VALID" :: ByteString
                SAME  -> "SAME"
            strides = case format of
                NCHW -> [1, 1, hStride, wStride]
                NHWC -> [1, hStride, wStride, 1]
            run x = activation $
                TF.conv2D' ( (TF.opAttr "strides" .~ strides)
                           . (TF.opAttr "padding" .~ paddingString)
                           . (TF.opAttr "data_format" .~ formatString)
                           . (TF.opAttr "use_cudnn_on_gpu" .~ True)
                           )
                        x (TF.readValue filter)
                `TF.add` TF.readValue bias
        let outSize inSize filterSize stride = case padding of
                -- Formulas documented in
                -- https://www.tensorflow.org/api_guides/python/nn#Convolution.
                VALID -> ceiling (fromIntegral (inSize - filterSize + 1)
                                  / fromIntegral stride)
                SAME  -> ceiling (fromIntegral inSize / fromIntegral stride)
            hOutSize = outSize h hFilterSize hStride
            wOutSize = outSize w wFilterSize wStride
            outShape = case format of
                NCHW -> [n, outputSize, hOutSize, wOutSize]
                NHWC -> [n, hOutSize, wOutSize, outputSize]
        pure ([filter, bias], outShape, TF.render . run)
