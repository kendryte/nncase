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

namespace Nncase.Transform.Pattern.Math
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

namespace Nncase.Transform.Pattern.NN
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

    public sealed record Conv2DPattern(Func<Conv2D, bool> Cond) : OpPattern
    {
        public Conv2DPattern(Conv2D conv2d): this(x => x == conv2d)
        {
        }

        public bool MatchLeaf(Conv2D conv2d) => Cond(conv2d) && MatchCheckedType(conv2d);
        public Conv2DPattern(): this((Conv2D x) => true)
        {
        }

        public Conv2DPattern(PadMode padMode): this((Conv2D x) => padMode == x.PadMode)
        {
        }
    }
}

namespace Nncase.Transform.Pattern.Tensors
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

    public sealed record PadPattern(Func<Pad, bool> Cond) : OpPattern
    {
        public PadPattern(Pad pad): this(x => x == pad)
        {
        }

        public bool MatchLeaf(Pad pad) => Cond(pad) && MatchCheckedType(pad);
        public PadPattern(): this((Pad x) => true)
        {
        }

        public PadPattern(PadMode padMode): this((Pad x) => padMode == x.PadMode)
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

        public ReducePattern(ReduceOp reduceOp): this((Reduce x) => reduceOp == x.ReduceOp)
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
}