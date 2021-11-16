using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.NN;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern.NN;
using Nncase.Transform.Pattern.Math;
using Nncase.Transform.Pattern.Tensors;

namespace Nncase.Transform.Pattern
{
    public abstract record OpPattern : ExprPattern
    {
        public bool MatchLeaf(Op op) => (this, op)switch
        {
        (BinaryPattern binarypattern, Binary binary) => binarypattern.MatchLeaf(binary), (ClampPattern clamppattern, Clamp clamp) => clamppattern.MatchLeaf(clamp), (UnaryPattern unarypattern, Unary unary) => unarypattern.MatchLeaf(unary), (SigmoidPattern sigmoidpattern, Sigmoid sigmoid) => sigmoidpattern.MatchLeaf(sigmoid), (ReluPattern relupattern, Relu relu) => relupattern.MatchLeaf(relu), (Relu6Pattern relu6pattern, Relu6 relu6) => relu6pattern.MatchLeaf(relu6), (PReluPattern prelupattern, PRelu prelu) => prelupattern.MatchLeaf(prelu), (LeakyReluPattern leakyrelupattern, LeakyRelu leakyrelu) => leakyrelupattern.MatchLeaf(leakyrelu), (Conv2DPattern conv2dpattern, Conv2D conv2d) => conv2dpattern.MatchLeaf(conv2d), (Conv2DTransposePattern conv2dtransposepattern, Conv2DTranspose conv2dtranspose) => conv2dtransposepattern.MatchLeaf(conv2dtranspose), (L2NormalizationPattern l2normalizationpattern, L2Normalization l2normalization) => l2normalizationpattern.MatchLeaf(l2normalization), (SoftMaxPattern softmaxpattern, SoftMax softmax) => softmaxpattern.MatchLeaf(softmax), (LogSoftMaxPattern logsoftmaxpattern, LogSoftMax logsoftmax) => logsoftmaxpattern.MatchLeaf(logsoftmax), (BatchToSpacePattern batchtospacepattern, BatchToSpace batchtospace) => batchtospacepattern.MatchLeaf(batchtospace), (BroadcastPattern broadcastpattern, Broadcast broadcast) => broadcastpattern.MatchLeaf(broadcast), (CastPattern castpattern, Cast cast) => castpattern.MatchLeaf(cast), (ConcatPattern concatpattern, Concat concat) => concatpattern.MatchLeaf(concat), (DeQuantizePattern dequantizepattern, DeQuantize dequantize) => dequantizepattern.MatchLeaf(dequantize), (GatherPattern gatherpattern, Gather gather) => gatherpattern.MatchLeaf(gather), (GatherNDPattern gatherndpattern, GatherND gathernd) => gatherndpattern.MatchLeaf(gathernd), (MatMulPattern matmulpattern, MatMul matmul) => matmulpattern.MatchLeaf(matmul), (OneHotPattern onehotpattern, OneHot onehot) => onehotpattern.MatchLeaf(onehot), (PadPattern padpattern, Pad pad) => padpattern.MatchLeaf(pad), (PaddingsPattern paddingspattern, Paddings paddings) => paddingspattern.MatchLeaf(paddings), (QuantizePattern quantizepattern, Quantize quantize) => quantizepattern.MatchLeaf(quantize), (ReducePattern reducepattern, Reduce reduce) => reducepattern.MatchLeaf(reduce), (ReduceWindow2DPattern reducewindow2dpattern, ReduceWindow2D reducewindow2d) => reducewindow2dpattern.MatchLeaf(reducewindow2d), (ReshapePattern reshapepattern, Reshape reshape) => reshapepattern.MatchLeaf(reshape), (ResizeImagePattern resizeimagepattern, ResizeImage resizeimage) => resizeimagepattern.MatchLeaf(resizeimage), (ShapeOpPattern shapeoppattern, ShapeOp shapeop) => shapeoppattern.MatchLeaf(shapeop), (SlicePattern slicepattern, Slice slice) => slicepattern.MatchLeaf(slice), (SpaceToBatchPattern spacetobatchpattern, SpaceToBatch spacetobatch) => spacetobatchpattern.MatchLeaf(spacetobatch), (SplitPattern splitpattern, Split split) => splitpattern.MatchLeaf(split), (SqueezePattern squeezepattern, Squeeze squeeze) => squeezepattern.MatchLeaf(squeeze), (StackPattern stackpattern, Stack stack) => stackpattern.MatchLeaf(stack), (TransposePattern transposepattern, Transpose transpose) => transposepattern.MatchLeaf(transpose), (UnSqueezePattern unsqueezepattern, UnSqueeze unsqueeze) => unsqueezepattern.MatchLeaf(unsqueeze), (_, _) => false
        }

        ;
    }
}