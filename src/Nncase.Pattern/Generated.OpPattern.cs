// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Pattern.NN;
using Nncase.Pattern.Math;
using Nncase.Pattern.Tensors;
using Nncase.IR.Random;
using Nncase.IR.Imaging;

namespace Nncase.Pattern
{
    public abstract partial record OpPattern : ExprPattern
    {
        public bool MatchLeaf(Op op) => (this, op) switch
        {
            (BinaryPattern binarypattern, Binary binary) => binarypattern.MatchLeaf(binary),
            (ClampPattern clamppattern, Clamp clamp) => clamppattern.MatchLeaf(clamp),
            (UnaryPattern unarypattern, Unary unary) => unarypattern.MatchLeaf(unary),
            (SigmoidPattern sigmoidpattern, Sigmoid sigmoid) => sigmoidpattern.MatchLeaf(sigmoid),
            (ReluPattern relupattern, Relu relu) => relupattern.MatchLeaf(relu),
            (Relu6Pattern relu6pattern, Relu6 relu6) => relu6pattern.MatchLeaf(relu6),
            (PReluPattern prelupattern, PRelu prelu) => prelupattern.MatchLeaf(prelu),
            (LeakyReluPattern leakyrelupattern, LeakyRelu leakyrelu) => leakyrelupattern.MatchLeaf(leakyrelu),
            (CeluPattern celupattern, Celu celu) => celupattern.MatchLeaf(celu),
            (SeluPattern selupattern, Selu selu) => selupattern.MatchLeaf(selu),
            (EluPattern elupattern, Elu elu) => elupattern.MatchLeaf(elu),
            (HardSwishPattern hardswishpattern, HardSwish hardswish) => hardswishpattern.MatchLeaf(hardswish),
            (HardSigmoidPattern hardsigmoidpattern, HardSigmoid hardsigmoid) => hardsigmoidpattern.MatchLeaf(hardsigmoid),
            (Conv2DPattern conv2dpattern, Conv2D conv2d) => conv2dpattern.MatchLeaf(conv2d),
            (Conv2DTransposePattern conv2dtransposepattern, Conv2DTranspose conv2dtranspose) => conv2dtransposepattern.MatchLeaf(conv2dtranspose),
            (L2NormalizationPattern l2normalizationpattern, L2Normalization l2normalization) => l2normalizationpattern.MatchLeaf(l2normalization),
            (BatchNormalizationPattern batchnormalizationpattern, BatchNormalization batchnormalization) => batchnormalizationpattern.MatchLeaf(batchnormalization),
            (InstanceNormalizationPattern instancenormalizationpattern, InstanceNormalization instancenormalization) => instancenormalizationpattern.MatchLeaf(instancenormalization),
            (LpNormalizationPattern lpnormalizationpattern, LpNormalization lpnormalization) => lpnormalizationpattern.MatchLeaf(lpnormalization),
            (LRNPattern lrnpattern, LRN lrn) => lrnpattern.MatchLeaf(lrn),
            (LogSoftMaxPattern logsoftmaxpattern, LogSoftmax logsoftmax) => logsoftmaxpattern.MatchLeaf(logsoftmax),
            (SoftMaxPattern softmaxpattern, Softmax softmax) => softmaxpattern.MatchLeaf(softmax),
            (SoftPlusPattern softpluspattern, Softplus softplus) => softpluspattern.MatchLeaf(softplus),
            (SoftSignPattern softsignpattern, Softsign softsign) => softsignpattern.MatchLeaf(softsign),
            (BatchToSpacePattern batchtospacepattern, BatchToSpace batchtospace) => batchtospacepattern.MatchLeaf(batchtospace),
            (BroadcastPattern broadcastpattern, Broadcast broadcast) => broadcastpattern.MatchLeaf(broadcast),
            (CastPattern castpattern, Cast cast) => castpattern.MatchLeaf(cast),
            (ConcatPattern concatpattern, Concat concat) => concatpattern.MatchLeaf(concat),
            (CumSumPattern cumsumpattern, CumSum cumsum) => cumsumpattern.MatchLeaf(cumsum),
            (DeQuantizePattern dequantizepattern, Dequantize dequantize) => dequantizepattern.MatchLeaf(dequantize),
            (GatherPattern gatherpattern, Gather gather) => gatherpattern.MatchLeaf(gather),
            (GatherNDPattern gatherndpattern, GatherND gathernd) => gatherndpattern.MatchLeaf(gathernd),
            (HardMaxPattern hardmaxpattern, Hardmax hardmax) => hardmaxpattern.MatchLeaf(hardmax),
            (MatMulPattern matmulpattern, MatMul matmul) => matmulpattern.MatchLeaf(matmul),
            (OneHotPattern onehotpattern, OneHot onehot) => onehotpattern.MatchLeaf(onehot),
            (PadPattern padpattern, Pad pad) => padpattern.MatchLeaf(pad),
            (QuantizePattern quantizepattern, Quantize quantize) => quantizepattern.MatchLeaf(quantize),
            (RandomNormalPattern randomnormalpattern, Normal randomnormal) => randomnormalpattern.MatchLeaf(randomnormal),
            (RandomNormalLikePattern randomnormallikepattern, NormalLike randomnormallike) => randomnormallikepattern.MatchLeaf(randomnormallike),
            (RandomUniformPattern randomuniformpattern, Uniform randomuniform) => randomuniformpattern.MatchLeaf(randomuniform),
            (RandomUniformLikePattern randomuniformlikepattern, UniformLike randomuniformlike) => randomuniformlikepattern.MatchLeaf(randomuniformlike),
            (ReducePattern reducepattern, Reduce reduce) => reducepattern.MatchLeaf(reduce),
            (ReduceArgPattern reduceargpattern, ReduceArg reducearg) => reduceargpattern.MatchLeaf(reducearg),
            (ReduceWindow2DPattern reducewindow2dpattern, ReduceWindow2D reducewindow2d) => reducewindow2dpattern.MatchLeaf(reducewindow2d),
            (ReshapePattern reshapepattern, Reshape reshape) => reshapepattern.MatchLeaf(reshape),
            (ResizeImagePattern resizeimagepattern, ResizeImage resizeimage) => resizeimagepattern.MatchLeaf(resizeimage),
            (ShapeOpPattern shapeoppattern, ShapeOf shapeop) => shapeoppattern.MatchLeaf(shapeop),
            (SlicePattern slicepattern, Slice slice) => slicepattern.MatchLeaf(slice),
            (SpaceToBatchPattern spacetobatchpattern, SpaceToBatch spacetobatch) => spacetobatchpattern.MatchLeaf(spacetobatch),
            (SplitPattern splitpattern, Split split) => splitpattern.MatchLeaf(split),
            (SqueezePattern squeezepattern, Squeeze squeeze) => squeezepattern.MatchLeaf(squeeze),
            (StackPattern stackpattern, Stack stack) => stackpattern.MatchLeaf(stack),
            (TransposePattern transposepattern, Transpose transpose) => transposepattern.MatchLeaf(transpose),
            (UnSqueezePattern unsqueezepattern, Unsqueeze unsqueeze) => unsqueezepattern.MatchLeaf(unsqueeze),
            (_, _) => false,
        }

        ;
        public static ExprPattern CastToPattern(Op op) => op switch
        {
            Binary binary => new BinaryPattern(binary),
            Clamp clamp => new ClampPattern(clamp),
            Unary unary => new UnaryPattern(unary),
            Sigmoid sigmoid => new SigmoidPattern(sigmoid),
            Relu relu => new ReluPattern(relu),
            Relu6 relu6 => new Relu6Pattern(relu6),
            PRelu prelu => new PReluPattern(prelu),
            LeakyRelu leakyrelu => new LeakyReluPattern(leakyrelu),
            Celu celu => new CeluPattern(celu),
            Selu selu => new SeluPattern(selu),
            Elu elu => new EluPattern(elu),
            HardSwish hardswish => new HardSwishPattern(hardswish),
            HardSigmoid hardsigmoid => new HardSigmoidPattern(hardsigmoid),
            Conv2D conv2d => new Conv2DPattern(conv2d),
            Conv2DTranspose conv2dtranspose => new Conv2DTransposePattern(conv2dtranspose),
            L2Normalization l2normalization => new L2NormalizationPattern(l2normalization),
            BatchNormalization batchnormalization => new BatchNormalizationPattern(batchnormalization),
            InstanceNormalization instancenormalization => new InstanceNormalizationPattern(instancenormalization),
            LpNormalization lpnormalization => new LpNormalizationPattern(lpnormalization),
            LRN lrn => new LRNPattern(lrn),
            LogSoftmax logsoftmax => new LogSoftMaxPattern(logsoftmax),
            Softmax softmax => new SoftMaxPattern(softmax),
            Softplus softplus => new SoftPlusPattern(softplus),
            Softsign softsign => new SoftSignPattern(softsign),
            BatchToSpace batchtospace => new BatchToSpacePattern(batchtospace),
            Broadcast broadcast => new BroadcastPattern(broadcast),
            Cast cast => new CastPattern(cast),
            Concat concat => new ConcatPattern(concat),
            CumSum cumsum => new CumSumPattern(cumsum),
            Dequantize dequantize => new DeQuantizePattern(dequantize),
            Gather gather => new GatherPattern(gather),
            GatherND gathernd => new GatherNDPattern(gathernd),
            Hardmax hardmax => new HardMaxPattern(hardmax),
            MatMul matmul => new MatMulPattern(matmul),
            OneHot onehot => new OneHotPattern(onehot),
            Pad pad => new PadPattern(pad),
            Quantize quantize => new QuantizePattern(quantize),
            Normal randomnormal => new RandomNormalPattern(randomnormal),
            NormalLike randomnormallike => new RandomNormalLikePattern(randomnormallike),
            Uniform randomuniform => new RandomUniformPattern(randomuniform),
            UniformLike randomuniformlike => new RandomUniformLikePattern(randomuniformlike),
            Reduce reduce => new ReducePattern(reduce),
            ReduceArg reducearg => new ReduceArgPattern(reducearg),
            ReduceWindow2D reducewindow2d => new ReduceWindow2DPattern(reducewindow2d),
            Reshape reshape => new ReshapePattern(reshape),
            ResizeImage resizeimage => new ResizeImagePattern(resizeimage),
            ShapeOf shapeop => new ShapeOpPattern(shapeop),
            Slice slice => new SlicePattern(slice),
            SpaceToBatch spacetobatch => new SpaceToBatchPattern(spacetobatch),
            Split split => new SplitPattern(split),
            Squeeze squeeze => new SqueezePattern(squeeze),
            Stack stack => new StackPattern(stack),
            Transpose transpose => new TransposePattern(transpose),
            Unsqueeze unsqueeze => new UnSqueezePattern(unsqueeze),
            _ => throw new NotImplementedException($"Can't Convert OP {op.GetType().Name} To ExprPattern"),
        }

        ;
    }
}