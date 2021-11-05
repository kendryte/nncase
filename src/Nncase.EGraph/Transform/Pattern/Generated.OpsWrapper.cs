using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;

namespace Nncase.Transform.Pattern.Math
{
    public sealed record BinaryWrapper(CallPattern Pattern)
    {
        public ExprPattern Lhs => Pattern[Binary.Lhs];
        public ExprPattern Rhs => Pattern[Binary.Rhs];
        public static implicit operator CallPattern(BinaryWrapper warper) => warper.Pattern;
    }

    public sealed record ClampWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Clamp.Input];
        public ExprPattern Min => Pattern[Clamp.Min];
        public ExprPattern Max => Pattern[Clamp.Max];
        public static implicit operator CallPattern(ClampWrapper warper) => warper.Pattern;
    }

    public sealed record UnaryWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Unary.Input];
        public static implicit operator CallPattern(UnaryWrapper warper) => warper.Pattern;
    }
}

namespace Nncase.Transform.Pattern.NN
{
    public sealed record SigmoidWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Sigmoid.Input];
        public static implicit operator CallPattern(SigmoidWrapper warper) => warper.Pattern;
    }
}

namespace Nncase.Transform.Pattern.Tensors
{
    public sealed record BatchToSpaceWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[BatchToSpace.Input];
        public ExprPattern BlockShape => Pattern[BatchToSpace.BlockShape];
        public ExprPattern Crops => Pattern[BatchToSpace.Crops];
        public static implicit operator CallPattern(BatchToSpaceWrapper warper) => warper.Pattern;
    }

    public sealed record BroadcastWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Broadcast.Input];
        public ExprPattern Shape => Pattern[Broadcast.Shape];
        public static implicit operator CallPattern(BroadcastWrapper warper) => warper.Pattern;
    }

    public sealed record CastWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Cast.Input];
        public static implicit operator CallPattern(CastWrapper warper) => warper.Pattern;
    }

    public sealed record ConcatWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Concat.Input];
        public ExprPattern Axis => Pattern[Concat.Axis];
        public static implicit operator CallPattern(ConcatWrapper warper) => warper.Pattern;
    }

    public sealed record Conv2DWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Conv2D.Input];
        public ExprPattern Weights => Pattern[Conv2D.Weights];
        public ExprPattern Bias => Pattern[Conv2D.Bias];
        public ExprPattern Stride => Pattern[Conv2D.Stride];
        public ExprPattern Padding => Pattern[Conv2D.Padding];
        public ExprPattern Dilation => Pattern[Conv2D.Dilation];
        public ExprPattern Groups => Pattern[Conv2D.Groups];
        public static implicit operator CallPattern(Conv2DWrapper warper) => warper.Pattern;
    }

    public sealed record DeQuantizeWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[DeQuantize.Input];
        public ExprPattern QuantParam => Pattern[DeQuantize.QuantParam];
        public static implicit operator CallPattern(DeQuantizeWrapper warper) => warper.Pattern;
    }

    public sealed record GatherWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Gather.Input];
        public ExprPattern Axis => Pattern[Gather.Axis];
        public ExprPattern Index => Pattern[Gather.Index];
        public static implicit operator CallPattern(GatherWrapper warper) => warper.Pattern;
    }

    public sealed record GatherNDWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[GatherND.Input];
        public ExprPattern Axis => Pattern[GatherND.Axis];
        public ExprPattern BatchDims => Pattern[GatherND.BatchDims];
        public ExprPattern Index => Pattern[GatherND.Index];
        public static implicit operator CallPattern(GatherNDWrapper warper) => warper.Pattern;
    }

    public sealed record MatMulWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[MatMul.Input];
        public ExprPattern Other => Pattern[MatMul.Other];
        public static implicit operator CallPattern(MatMulWrapper warper) => warper.Pattern;
    }

    public sealed record PadWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Pad.Input];
        public ExprPattern Pads => Pattern[Pad.Pads];
        public ExprPattern Value => Pattern[Pad.Value];
        public static implicit operator CallPattern(PadWrapper warper) => warper.Pattern;
    }

    public sealed record QuantizeWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Quantize.Input];
        public ExprPattern QuantParam => Pattern[Quantize.QuantParam];
        public static implicit operator CallPattern(QuantizeWrapper warper) => warper.Pattern;
    }

    public sealed record ReduceWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Reduce.Input];
        public ExprPattern Axis => Pattern[Reduce.Axis];
        public ExprPattern InitValue => Pattern[Reduce.InitValue];
        public ExprPattern KeepDims => Pattern[Reduce.KeepDims];
        public static implicit operator CallPattern(ReduceWrapper warper) => warper.Pattern;
    }

    public sealed record ReshapeWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Reshape.Input];
        public ExprPattern Shape => Pattern[Reshape.Shape];
        public static implicit operator CallPattern(ReshapeWrapper warper) => warper.Pattern;
    }

    public sealed record SliceWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Slice.Input];
        public ExprPattern Begins => Pattern[Slice.Begins];
        public ExprPattern Ends => Pattern[Slice.Ends];
        public ExprPattern Axes => Pattern[Slice.Axes];
        public ExprPattern Strides => Pattern[Slice.Strides];
        public static implicit operator CallPattern(SliceWrapper warper) => warper.Pattern;
    }

    public sealed record SpaceToBatchWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[SpaceToBatch.Input];
        public ExprPattern BlockShape => Pattern[SpaceToBatch.BlockShape];
        public ExprPattern Paddings => Pattern[SpaceToBatch.Paddings];
        public static implicit operator CallPattern(SpaceToBatchWrapper warper) => warper.Pattern;
    }

    public sealed record SplitWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Split.Input];
        public ExprPattern Axis => Pattern[Split.Axis];
        public ExprPattern Sections => Pattern[Split.Sections];
        public static implicit operator CallPattern(SplitWrapper warper) => warper.Pattern;
    }

    public sealed record SqueezeWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Squeeze.Input];
        public ExprPattern Dims => Pattern[Squeeze.Dims];
        public static implicit operator CallPattern(SqueezeWrapper warper) => warper.Pattern;
    }

    public sealed record TransposeWrapper(CallPattern Pattern)
    {
        public ExprPattern Input => Pattern[Transpose.Input];
        public ExprPattern Perm => Pattern[Transpose.Perm];
        public static implicit operator CallPattern(TransposeWrapper warper) => warper.Pattern;
    }
}