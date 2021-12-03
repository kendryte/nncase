using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;

namespace Nncase.Pattern.Math
{
    public sealed record BinaryPattern(Func<Binary, bool> Cond) : OpPattern
    {
        public BinaryPattern(Binary binary): this(x => x == binary)
        {
        }

        public bool MatchLeaf(Binary binary) => Cond(binary) && MatchCheckedType(binary);
        public BinaryPattern(): this((Binary x) => true)
        {
        }

        public BinaryPattern(BinaryOp BinaryOp): this((Binary x) => BinaryOp == x.BinaryOp)
        {
        }
    }

    public sealed record ClampPattern(Func<Clamp, bool> Cond) : OpPattern
    {
        public ClampPattern(Clamp clamp): this(x => x == clamp)
        {
        }

        public bool MatchLeaf(Clamp clamp) => Cond(clamp) && MatchCheckedType(clamp);
        public ClampPattern(): this((Clamp x) => true)
        {
        }
    }

    public sealed record UnaryPattern(Func<Unary, bool> Cond) : OpPattern
    {
        public UnaryPattern(Unary unary): this(x => x == unary)
        {
        }

        public bool MatchLeaf(Unary unary) => Cond(unary) && MatchCheckedType(unary);
        public UnaryPattern(): this((Unary x) => true)
        {
        }

        public UnaryPattern(UnaryOp UnaryOp): this((Unary x) => UnaryOp == x.UnaryOp)
        {
        }
    }
}

namespace Nncase.Pattern.NN
{
    public sealed record SigmoidPattern(Func<Sigmoid, bool> Cond) : OpPattern
    {
        public SigmoidPattern(Sigmoid sigmoid): this(x => x == sigmoid)
        {
        }

        public bool MatchLeaf(Sigmoid sigmoid) => Cond(sigmoid) && MatchCheckedType(sigmoid);
        public SigmoidPattern(): this((Sigmoid x) => true)
        {
        }
    }

    public sealed record ReluPattern(Func<Relu, bool> Cond) : OpPattern
    {
        public ReluPattern(Relu relu): this(x => x == relu)
        {
        }

        public bool MatchLeaf(Relu relu) => Cond(relu) && MatchCheckedType(relu);
        public ReluPattern(): this((Relu x) => true)
        {
        }
    }

    public sealed record Relu6Pattern(Func<Relu6, bool> Cond) : OpPattern
    {
        public Relu6Pattern(Relu6 relu6): this(x => x == relu6)
        {
        }

        public bool MatchLeaf(Relu6 relu6) => Cond(relu6) && MatchCheckedType(relu6);
        public Relu6Pattern(): this((Relu6 x) => true)
        {
        }
    }

    public sealed record PReluPattern(Func<PRelu, bool> Cond) : OpPattern
    {
        public PReluPattern(PRelu prelu): this(x => x == prelu)
        {
        }

        public bool MatchLeaf(PRelu prelu) => Cond(prelu) && MatchCheckedType(prelu);
        public PReluPattern(): this((PRelu x) => true)
        {
        }
    }

    public sealed record LeakyReluPattern(Func<LeakyRelu, bool> Cond) : OpPattern
    {
        public LeakyReluPattern(LeakyRelu leakyrelu): this(x => x == leakyrelu)
        {
        }

        public bool MatchLeaf(LeakyRelu leakyrelu) => Cond(leakyrelu) && MatchCheckedType(leakyrelu);
        public LeakyReluPattern(): this((LeakyRelu x) => true)
        {
        }
    }

    public sealed record Conv2DPattern(Func<Conv2D, bool> Cond) : OpPattern
    {
        public Conv2DPattern(Conv2D conv2d): this(x => x == conv2d)
        {
        }

        public bool MatchLeaf(Conv2D conv2d) => Cond(conv2d) && MatchCheckedType(conv2d);
        public Conv2DPattern(): this((Conv2D x) => true)
        {
        }

        public Conv2DPattern(PadMode PadMode): this((Conv2D x) => PadMode == x.PadMode)
        {
        }
    }

    public sealed record Conv2DTransposePattern(Func<Conv2DTranspose, bool> Cond) : OpPattern
    {
        public Conv2DTransposePattern(Conv2DTranspose conv2dtranspose): this(x => x == conv2dtranspose)
        {
        }

        public bool MatchLeaf(Conv2DTranspose conv2dtranspose) => Cond(conv2dtranspose) && MatchCheckedType(conv2dtranspose);
        public Conv2DTransposePattern(): this((Conv2DTranspose x) => true)
        {
        }

        public Conv2DTransposePattern(PadMode PadMode): this((Conv2DTranspose x) => PadMode == x.PadMode)
        {
        }
    }

    public sealed record L2NormalizationPattern(Func<L2Normalization, bool> Cond) : OpPattern
    {
        public L2NormalizationPattern(L2Normalization l2normalization): this(x => x == l2normalization)
        {
        }

        public bool MatchLeaf(L2Normalization l2normalization) => Cond(l2normalization) && MatchCheckedType(l2normalization);
        public L2NormalizationPattern(): this((L2Normalization x) => true)
        {
        }
    }

    public sealed record BatchNormalizationPattern(Func<BatchNormalization, bool> Cond) : OpPattern
    {
        public BatchNormalizationPattern(BatchNormalization batchnormalization): this(x => x == batchnormalization)
        {
        }

        public bool MatchLeaf(BatchNormalization batchnormalization) => Cond(batchnormalization) && MatchCheckedType(batchnormalization);
        public BatchNormalizationPattern(): this((BatchNormalization x) => true)
        {
        }
    }

    public sealed record InstanceNormalizationPattern(Func<InstanceNormalization, bool> Cond) : OpPattern
    {
        public InstanceNormalizationPattern(InstanceNormalization instancenormalization): this(x => x == instancenormalization)
        {
        }

        public bool MatchLeaf(InstanceNormalization instancenormalization) => Cond(instancenormalization) && MatchCheckedType(instancenormalization);
        public InstanceNormalizationPattern(): this((InstanceNormalization x) => true)
        {
        }
    }

    public sealed record LpNormalizationPattern(Func<LpNormalization, bool> Cond) : OpPattern
    {
        public LpNormalizationPattern(LpNormalization lpnormalization): this(x => x == lpnormalization)
        {
        }

        public bool MatchLeaf(LpNormalization lpnormalization) => Cond(lpnormalization) && MatchCheckedType(lpnormalization);
        public LpNormalizationPattern(): this((LpNormalization x) => true)
        {
        }
    }

    public sealed record SoftMaxPattern(Func<SoftMax, bool> Cond) : OpPattern
    {
        public SoftMaxPattern(SoftMax softmax): this(x => x == softmax)
        {
        }

        public bool MatchLeaf(SoftMax softmax) => Cond(softmax) && MatchCheckedType(softmax);
        public SoftMaxPattern(): this((SoftMax x) => true)
        {
        }
    }

    public sealed record LogSoftMaxPattern(Func<LogSoftMax, bool> Cond) : OpPattern
    {
        public LogSoftMaxPattern(LogSoftMax logsoftmax): this(x => x == logsoftmax)
        {
        }

        public bool MatchLeaf(LogSoftMax logsoftmax) => Cond(logsoftmax) && MatchCheckedType(logsoftmax);
        public LogSoftMaxPattern(): this((LogSoftMax x) => true)
        {
        }
    }
}

namespace Nncase.Pattern.Tensors
{
    public sealed record BatchToSpacePattern(Func<BatchToSpace, bool> Cond) : OpPattern
    {
        public BatchToSpacePattern(BatchToSpace batchtospace): this(x => x == batchtospace)
        {
        }

        public bool MatchLeaf(BatchToSpace batchtospace) => Cond(batchtospace) && MatchCheckedType(batchtospace);
        public BatchToSpacePattern(): this((BatchToSpace x) => true)
        {
        }
    }

    public sealed record BroadcastPattern(Func<Broadcast, bool> Cond) : OpPattern
    {
        public BroadcastPattern(Broadcast broadcast): this(x => x == broadcast)
        {
        }

        public bool MatchLeaf(Broadcast broadcast) => Cond(broadcast) && MatchCheckedType(broadcast);
        public BroadcastPattern(): this((Broadcast x) => true)
        {
        }
    }

    public sealed record CastPattern(Func<Cast, bool> Cond) : OpPattern
    {
        public CastPattern(Cast cast): this(x => x == cast)
        {
        }

        public bool MatchLeaf(Cast cast) => Cond(cast) && MatchCheckedType(cast);
        public CastPattern(): this((Cast x) => true)
        {
        }

        public CastPattern(DataType NewType): this((Cast x) => NewType == x.NewType)
        {
        }
    }

    public sealed record ConcatPattern(Func<Concat, bool> Cond) : OpPattern
    {
        public ConcatPattern(Concat concat): this(x => x == concat)
        {
        }

        public bool MatchLeaf(Concat concat) => Cond(concat) && MatchCheckedType(concat);
        public ConcatPattern(): this((Concat x) => true)
        {
        }
    }

    public sealed record DeQuantizePattern(Func<DeQuantize, bool> Cond) : OpPattern
    {
        public DeQuantizePattern(DeQuantize dequantize): this(x => x == dequantize)
        {
        }

        public bool MatchLeaf(DeQuantize dequantize) => Cond(dequantize) && MatchCheckedType(dequantize);
        public DeQuantizePattern(): this((DeQuantize x) => true)
        {
        }

        public DeQuantizePattern(DataType TargetType): this((DeQuantize x) => TargetType == x.TargetType)
        {
        }
    }

    public sealed record GatherPattern(Func<Gather, bool> Cond) : OpPattern
    {
        public GatherPattern(Gather gather): this(x => x == gather)
        {
        }

        public bool MatchLeaf(Gather gather) => Cond(gather) && MatchCheckedType(gather);
        public GatherPattern(): this((Gather x) => true)
        {
        }
    }

    public sealed record GatherNDPattern(Func<GatherND, bool> Cond) : OpPattern
    {
        public GatherNDPattern(GatherND gathernd): this(x => x == gathernd)
        {
        }

        public bool MatchLeaf(GatherND gathernd) => Cond(gathernd) && MatchCheckedType(gathernd);
        public GatherNDPattern(): this((GatherND x) => true)
        {
        }
    }

    public sealed record MatMulPattern(Func<MatMul, bool> Cond) : OpPattern
    {
        public MatMulPattern(MatMul matmul): this(x => x == matmul)
        {
        }

        public bool MatchLeaf(MatMul matmul) => Cond(matmul) && MatchCheckedType(matmul);
        public MatMulPattern(): this((MatMul x) => true)
        {
        }
    }

    public sealed record OneHotPattern(Func<OneHot, bool> Cond) : OpPattern
    {
        public OneHotPattern(OneHot onehot): this(x => x == onehot)
        {
        }

        public bool MatchLeaf(OneHot onehot) => Cond(onehot) && MatchCheckedType(onehot);
        public OneHotPattern(): this((OneHot x) => true)
        {
        }

        public OneHotPattern(OneHotMode OneHotMode): this((OneHot x) => OneHotMode == x.OneHotMode)
        {
        }
    }

    public sealed record PadPattern(Func<Pad, bool> Cond) : OpPattern
    {
        public PadPattern(Pad pad): this(x => x == pad)
        {
        }

        public bool MatchLeaf(Pad pad) => Cond(pad) && MatchCheckedType(pad);
        public PadPattern(): this((Pad x) => true)
        {
        }

        public PadPattern(PadMode PadMode): this((Pad x) => PadMode == x.PadMode)
        {
        }
    }

    public sealed record QuantizePattern(Func<Quantize, bool> Cond) : OpPattern
    {
        public QuantizePattern(Quantize quantize): this(x => x == quantize)
        {
        }

        public bool MatchLeaf(Quantize quantize) => Cond(quantize) && MatchCheckedType(quantize);
        public QuantizePattern(): this((Quantize x) => true)
        {
        }

        public QuantizePattern(DataType TargetType): this((Quantize x) => TargetType == x.TargetType)
        {
        }
    }

    public sealed record ReducePattern(Func<Reduce, bool> Cond) : OpPattern
    {
        public ReducePattern(Reduce reduce): this(x => x == reduce)
        {
        }

        public bool MatchLeaf(Reduce reduce) => Cond(reduce) && MatchCheckedType(reduce);
        public ReducePattern(): this((Reduce x) => true)
        {
        }

        public ReducePattern(ReduceOp ReduceOp): this((Reduce x) => ReduceOp == x.ReduceOp)
        {
        }
    }

    public sealed record ReduceArgPattern(Func<ReduceArg, bool> Cond) : OpPattern
    {
        public ReduceArgPattern(ReduceArg reducearg): this(x => x == reducearg)
        {
        }

        public bool MatchLeaf(ReduceArg reducearg) => Cond(reducearg) && MatchCheckedType(reducearg);
        public ReduceArgPattern(): this((ReduceArg x) => true)
        {
        }

        public ReduceArgPattern(ReduceArgOp ReduceArgOp): this((ReduceArg x) => ReduceArgOp == x.ReduceArgOp)
        {
        }
    }

    public sealed record ReduceWindow2DPattern(Func<ReduceWindow2D, bool> Cond) : OpPattern
    {
        public ReduceWindow2DPattern(ReduceWindow2D reducewindow2d): this(x => x == reducewindow2d)
        {
        }

        public bool MatchLeaf(ReduceWindow2D reducewindow2d) => Cond(reducewindow2d) && MatchCheckedType(reducewindow2d);
        public ReduceWindow2DPattern(): this((ReduceWindow2D x) => true)
        {
        }

        public ReduceWindow2DPattern(ReduceOp ReduceOp): this((ReduceWindow2D x) => ReduceOp == x.ReduceOp)
        {
        }
    }

    public sealed record ReshapePattern(Func<Reshape, bool> Cond) : OpPattern
    {
        public ReshapePattern(Reshape reshape): this(x => x == reshape)
        {
        }

        public bool MatchLeaf(Reshape reshape) => Cond(reshape) && MatchCheckedType(reshape);
        public ReshapePattern(): this((Reshape x) => true)
        {
        }
    }

    public sealed record ResizeImagePattern(Func<ResizeImage, bool> Cond) : OpPattern
    {
        public ResizeImagePattern(ResizeImage resizeimage): this(x => x == resizeimage)
        {
        }

        public bool MatchLeaf(ResizeImage resizeimage) => Cond(resizeimage) && MatchCheckedType(resizeimage);
        public ResizeImagePattern(): this((ResizeImage x) => true)
        {
        }

        public ResizeImagePattern(ImageResizeMode ResizeMode): this((ResizeImage x) => ResizeMode == x.ResizeMode)
        {
        }
    }

    public sealed record ShapeOpPattern(Func<ShapeOp, bool> Cond) : OpPattern
    {
        public ShapeOpPattern(ShapeOp shapeop): this(x => x == shapeop)
        {
        }

        public bool MatchLeaf(ShapeOp shapeop) => Cond(shapeop) && MatchCheckedType(shapeop);
        public ShapeOpPattern(): this((ShapeOp x) => true)
        {
        }
    }

    public sealed record SlicePattern(Func<Slice, bool> Cond) : OpPattern
    {
        public SlicePattern(Slice slice): this(x => x == slice)
        {
        }

        public bool MatchLeaf(Slice slice) => Cond(slice) && MatchCheckedType(slice);
        public SlicePattern(): this((Slice x) => true)
        {
        }
    }

    public sealed record SpaceToBatchPattern(Func<SpaceToBatch, bool> Cond) : OpPattern
    {
        public SpaceToBatchPattern(SpaceToBatch spacetobatch): this(x => x == spacetobatch)
        {
        }

        public bool MatchLeaf(SpaceToBatch spacetobatch) => Cond(spacetobatch) && MatchCheckedType(spacetobatch);
        public SpaceToBatchPattern(): this((SpaceToBatch x) => true)
        {
        }
    }

    public sealed record SplitPattern(Func<Split, bool> Cond) : OpPattern
    {
        public SplitPattern(Split split): this(x => x == split)
        {
        }

        public bool MatchLeaf(Split split) => Cond(split) && MatchCheckedType(split);
        public SplitPattern(): this((Split x) => true)
        {
        }
    }

    public sealed record SqueezePattern(Func<Squeeze, bool> Cond) : OpPattern
    {
        public SqueezePattern(Squeeze squeeze): this(x => x == squeeze)
        {
        }

        public bool MatchLeaf(Squeeze squeeze) => Cond(squeeze) && MatchCheckedType(squeeze);
        public SqueezePattern(): this((Squeeze x) => true)
        {
        }
    }

    public sealed record StackPattern(Func<Stack, bool> Cond) : OpPattern
    {
        public StackPattern(Stack stack): this(x => x == stack)
        {
        }

        public bool MatchLeaf(Stack stack) => Cond(stack) && MatchCheckedType(stack);
        public StackPattern(): this((Stack x) => true)
        {
        }
    }

    public sealed record TransposePattern(Func<Transpose, bool> Cond) : OpPattern
    {
        public TransposePattern(Transpose transpose): this(x => x == transpose)
        {
        }

        public bool MatchLeaf(Transpose transpose) => Cond(transpose) && MatchCheckedType(transpose);
        public TransposePattern(): this((Transpose x) => true)
        {
        }
    }

    public sealed record UnSqueezePattern(Func<UnSqueeze, bool> Cond) : OpPattern
    {
        public UnSqueezePattern(UnSqueeze unsqueeze): this(x => x == unsqueeze)
        {
        }

        public bool MatchLeaf(UnSqueeze unsqueeze) => Cond(unsqueeze) && MatchCheckedType(unsqueeze);
        public UnSqueezePattern(): this((UnSqueeze x) => true)
        {
        }
    }
}