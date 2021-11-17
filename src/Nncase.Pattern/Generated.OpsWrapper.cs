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
    public sealed record BinaryWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern LhsPat() => Pattern[Binary.Lhs];
        public T LhsPat<T>()
            where T : ExprPattern => (T)LhsPat();
        public Expr Lhs() => GetCast<Expr>(LhsPat());
        public T Lhs<T>()
            where T : Expr => GetCast<T>(LhsPat());
        public ExprPattern RhsPat() => Pattern[Binary.Rhs];
        public T RhsPat<T>()
            where T : ExprPattern => (T)RhsPat();
        public Expr Rhs() => GetCast<Expr>(RhsPat());
        public T Rhs<T>()
            where T : Expr => GetCast<T>(RhsPat());
        public BinaryOp BinaryOp => ((Binary)GetCast<Call>(this).Target).BinaryOp;
        public static implicit operator CallPattern(BinaryWrapper warper) => warper.Pattern;
    }

    public sealed record ClampWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Clamp.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern MinPat() => Pattern[Clamp.Min];
        public T MinPat<T>()
            where T : ExprPattern => (T)MinPat();
        public Expr Min() => GetCast<Expr>(MinPat());
        public T Min<T>()
            where T : Expr => GetCast<T>(MinPat());
        public ExprPattern MaxPat() => Pattern[Clamp.Max];
        public T MaxPat<T>()
            where T : ExprPattern => (T)MaxPat();
        public Expr Max() => GetCast<Expr>(MaxPat());
        public T Max<T>()
            where T : Expr => GetCast<T>(MaxPat());
        public static implicit operator CallPattern(ClampWrapper warper) => warper.Pattern;
    }

    public sealed record UnaryWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Unary.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public UnaryOp UnaryOp => ((Unary)GetCast<Call>(this).Target).UnaryOp;
        public static implicit operator CallPattern(UnaryWrapper warper) => warper.Pattern;
    }
}

namespace Nncase.Pattern.NN
{
    public sealed record SigmoidWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Sigmoid.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(SigmoidWrapper warper) => warper.Pattern;
    }

    public sealed record ReluWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Relu.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(ReluWrapper warper) => warper.Pattern;
    }

    public sealed record Relu6Wrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Relu6.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(Relu6Wrapper warper) => warper.Pattern;
    }

    public sealed record PReluWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[PRelu.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(PReluWrapper warper) => warper.Pattern;
    }

    public sealed record LeakyReluWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[LeakyRelu.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(LeakyReluWrapper warper) => warper.Pattern;
    }

    public sealed record Conv2DWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Conv2D.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern WeightsPat() => Pattern[Conv2D.Weights];
        public T WeightsPat<T>()
            where T : ExprPattern => (T)WeightsPat();
        public Expr Weights() => GetCast<Expr>(WeightsPat());
        public T Weights<T>()
            where T : Expr => GetCast<T>(WeightsPat());
        public ExprPattern BiasPat() => Pattern[Conv2D.Bias];
        public T BiasPat<T>()
            where T : ExprPattern => (T)BiasPat();
        public Expr Bias() => GetCast<Expr>(BiasPat());
        public T Bias<T>()
            where T : Expr => GetCast<T>(BiasPat());
        public ExprPattern StridePat() => Pattern[Conv2D.Stride];
        public T StridePat<T>()
            where T : ExprPattern => (T)StridePat();
        public Expr Stride() => GetCast<Expr>(StridePat());
        public T Stride<T>()
            where T : Expr => GetCast<T>(StridePat());
        public ExprPattern PaddingPat() => Pattern[Conv2D.Padding];
        public T PaddingPat<T>()
            where T : ExprPattern => (T)PaddingPat();
        public Expr Padding() => GetCast<Expr>(PaddingPat());
        public T Padding<T>()
            where T : Expr => GetCast<T>(PaddingPat());
        public ExprPattern DilationPat() => Pattern[Conv2D.Dilation];
        public T DilationPat<T>()
            where T : ExprPattern => (T)DilationPat();
        public Expr Dilation() => GetCast<Expr>(DilationPat());
        public T Dilation<T>()
            where T : Expr => GetCast<T>(DilationPat());
        public ExprPattern GroupsPat() => Pattern[Conv2D.Groups];
        public T GroupsPat<T>()
            where T : ExprPattern => (T)GroupsPat();
        public Expr Groups() => GetCast<Expr>(GroupsPat());
        public T Groups<T>()
            where T : Expr => GetCast<T>(GroupsPat());
        public PadMode PadMode => ((Conv2D)GetCast<Call>(this).Target).PadMode;
        public static implicit operator CallPattern(Conv2DWrapper warper) => warper.Pattern;
    }

    public sealed record Conv2DTransposeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Conv2DTranspose.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern WeightsPat() => Pattern[Conv2DTranspose.Weights];
        public T WeightsPat<T>()
            where T : ExprPattern => (T)WeightsPat();
        public Expr Weights() => GetCast<Expr>(WeightsPat());
        public T Weights<T>()
            where T : Expr => GetCast<T>(WeightsPat());
        public ExprPattern BiasPat() => Pattern[Conv2DTranspose.Bias];
        public T BiasPat<T>()
            where T : ExprPattern => (T)BiasPat();
        public Expr Bias() => GetCast<Expr>(BiasPat());
        public T Bias<T>()
            where T : Expr => GetCast<T>(BiasPat());
        public ExprPattern OutShapePat() => Pattern[Conv2DTranspose.OutShape];
        public T OutShapePat<T>()
            where T : ExprPattern => (T)OutShapePat();
        public Expr OutShape() => GetCast<Expr>(OutShapePat());
        public T OutShape<T>()
            where T : Expr => GetCast<T>(OutShapePat());
        public ExprPattern StridePat() => Pattern[Conv2DTranspose.Stride];
        public T StridePat<T>()
            where T : ExprPattern => (T)StridePat();
        public Expr Stride() => GetCast<Expr>(StridePat());
        public T Stride<T>()
            where T : Expr => GetCast<T>(StridePat());
        public ExprPattern PaddingPat() => Pattern[Conv2DTranspose.Padding];
        public T PaddingPat<T>()
            where T : ExprPattern => (T)PaddingPat();
        public Expr Padding() => GetCast<Expr>(PaddingPat());
        public T Padding<T>()
            where T : Expr => GetCast<T>(PaddingPat());
        public ExprPattern DilationPat() => Pattern[Conv2DTranspose.Dilation];
        public T DilationPat<T>()
            where T : ExprPattern => (T)DilationPat();
        public Expr Dilation() => GetCast<Expr>(DilationPat());
        public T Dilation<T>()
            where T : Expr => GetCast<T>(DilationPat());
        public ExprPattern GroupsPat() => Pattern[Conv2DTranspose.Groups];
        public T GroupsPat<T>()
            where T : ExprPattern => (T)GroupsPat();
        public Expr Groups() => GetCast<Expr>(GroupsPat());
        public T Groups<T>()
            where T : Expr => GetCast<T>(GroupsPat());
        public PadMode PadMode => ((Conv2DTranspose)GetCast<Call>(this).Target).PadMode;
        public static implicit operator CallPattern(Conv2DTransposeWrapper warper) => warper.Pattern;
    }

    public sealed record L2NormalizationWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[L2Normalization.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(L2NormalizationWrapper warper) => warper.Pattern;
    }

    public sealed record SoftMaxWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[SoftMax.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(SoftMaxWrapper warper) => warper.Pattern;
    }

    public sealed record LogSoftMaxWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[LogSoftMax.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(LogSoftMaxWrapper warper) => warper.Pattern;
    }
}

namespace Nncase.Pattern.Tensors
{
    public sealed record BatchToSpaceWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[BatchToSpace.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern BlockShapePat() => Pattern[BatchToSpace.BlockShape];
        public T BlockShapePat<T>()
            where T : ExprPattern => (T)BlockShapePat();
        public Expr BlockShape() => GetCast<Expr>(BlockShapePat());
        public T BlockShape<T>()
            where T : Expr => GetCast<T>(BlockShapePat());
        public ExprPattern CropsPat() => Pattern[BatchToSpace.Crops];
        public T CropsPat<T>()
            where T : ExprPattern => (T)CropsPat();
        public Expr Crops() => GetCast<Expr>(CropsPat());
        public T Crops<T>()
            where T : Expr => GetCast<T>(CropsPat());
        public static implicit operator CallPattern(BatchToSpaceWrapper warper) => warper.Pattern;
    }

    public sealed record BroadcastWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Broadcast.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern ShapePat() => Pattern[Broadcast.Shape];
        public T ShapePat<T>()
            where T : ExprPattern => (T)ShapePat();
        public Expr Shape() => GetCast<Expr>(ShapePat());
        public T Shape<T>()
            where T : Expr => GetCast<T>(ShapePat());
        public static implicit operator CallPattern(BroadcastWrapper warper) => warper.Pattern;
    }

    public sealed record CastWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Cast.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public DataType NewType => ((Cast)GetCast<Call>(this).Target).NewType;
        public static implicit operator CallPattern(CastWrapper warper) => warper.Pattern;
    }

    public sealed record ConcatWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Concat.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern AxisPat() => Pattern[Concat.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public static implicit operator CallPattern(ConcatWrapper warper) => warper.Pattern;
    }

    public sealed record DeQuantizeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[DeQuantize.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern ZeroPointPat() => Pattern[DeQuantize.ZeroPoint];
        public T ZeroPointPat<T>()
            where T : ExprPattern => (T)ZeroPointPat();
        public Expr ZeroPoint() => GetCast<Expr>(ZeroPointPat());
        public T ZeroPoint<T>()
            where T : Expr => GetCast<T>(ZeroPointPat());
        public ExprPattern ScalePat() => Pattern[DeQuantize.Scale];
        public T ScalePat<T>()
            where T : ExprPattern => (T)ScalePat();
        public Expr Scale() => GetCast<Expr>(ScalePat());
        public T Scale<T>()
            where T : Expr => GetCast<T>(ScalePat());
        public DataType TargetType => ((DeQuantize)GetCast<Call>(this).Target).TargetType;
        public static implicit operator CallPattern(DeQuantizeWrapper warper) => warper.Pattern;
    }

    public sealed record GatherWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Gather.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern AxisPat() => Pattern[Gather.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public ExprPattern IndexPat() => Pattern[Gather.Index];
        public T IndexPat<T>()
            where T : ExprPattern => (T)IndexPat();
        public Expr Index() => GetCast<Expr>(IndexPat());
        public T Index<T>()
            where T : Expr => GetCast<T>(IndexPat());
        public static implicit operator CallPattern(GatherWrapper warper) => warper.Pattern;
    }

    public sealed record GatherNDWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[GatherND.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern BatchDimsPat() => Pattern[GatherND.BatchDims];
        public T BatchDimsPat<T>()
            where T : ExprPattern => (T)BatchDimsPat();
        public Expr BatchDims() => GetCast<Expr>(BatchDimsPat());
        public T BatchDims<T>()
            where T : Expr => GetCast<T>(BatchDimsPat());
        public ExprPattern IndexPat() => Pattern[GatherND.Index];
        public T IndexPat<T>()
            where T : ExprPattern => (T)IndexPat();
        public Expr Index() => GetCast<Expr>(IndexPat());
        public T Index<T>()
            where T : Expr => GetCast<T>(IndexPat());
        public ExprPattern AxisPat() => Pattern[GatherND.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public static implicit operator CallPattern(GatherNDWrapper warper) => warper.Pattern;
    }

    public sealed record MatMulWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[MatMul.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern OtherPat() => Pattern[MatMul.Other];
        public T OtherPat<T>()
            where T : ExprPattern => (T)OtherPat();
        public Expr Other() => GetCast<Expr>(OtherPat());
        public T Other<T>()
            where T : Expr => GetCast<T>(OtherPat());
        public static implicit operator CallPattern(MatMulWrapper warper) => warper.Pattern;
    }

    public sealed record OneHotWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern IndicesPat() => Pattern[OneHot.Indices];
        public T IndicesPat<T>()
            where T : ExprPattern => (T)IndicesPat();
        public Expr Indices() => GetCast<Expr>(IndicesPat());
        public T Indices<T>()
            where T : Expr => GetCast<T>(IndicesPat());
        public ExprPattern DepthPat() => Pattern[OneHot.Depth];
        public T DepthPat<T>()
            where T : ExprPattern => (T)DepthPat();
        public Expr Depth() => GetCast<Expr>(DepthPat());
        public T Depth<T>()
            where T : Expr => GetCast<T>(DepthPat());
        public ExprPattern OnValuePat() => Pattern[OneHot.OnValue];
        public T OnValuePat<T>()
            where T : ExprPattern => (T)OnValuePat();
        public Expr OnValue() => GetCast<Expr>(OnValuePat());
        public T OnValue<T>()
            where T : Expr => GetCast<T>(OnValuePat());
        public ExprPattern OffValuePat() => Pattern[OneHot.OffValue];
        public T OffValuePat<T>()
            where T : ExprPattern => (T)OffValuePat();
        public Expr OffValue() => GetCast<Expr>(OffValuePat());
        public T OffValue<T>()
            where T : Expr => GetCast<T>(OffValuePat());
        public ExprPattern AxisPat() => Pattern[OneHot.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public OneHotMode OneHotMode => ((OneHot)GetCast<Call>(this).Target).OneHotMode;
        public static implicit operator CallPattern(OneHotWrapper warper) => warper.Pattern;
    }

    public sealed record PadWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Pad.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern PadsPat() => Pattern[Pad.Pads];
        public T PadsPat<T>()
            where T : ExprPattern => (T)PadsPat();
        public Expr Pads() => GetCast<Expr>(PadsPat());
        public T Pads<T>()
            where T : Expr => GetCast<T>(PadsPat());
        public ExprPattern ValuePat() => Pattern[Pad.Value];
        public T ValuePat<T>()
            where T : ExprPattern => (T)ValuePat();
        public Expr Value() => GetCast<Expr>(ValuePat());
        public T Value<T>()
            where T : Expr => GetCast<T>(ValuePat());
        public PadMode PadMode => ((Pad)GetCast<Call>(this).Target).PadMode;
        public static implicit operator CallPattern(PadWrapper warper) => warper.Pattern;
    }

    public sealed record QuantizeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Quantize.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern ZeroPointPat() => Pattern[Quantize.ZeroPoint];
        public T ZeroPointPat<T>()
            where T : ExprPattern => (T)ZeroPointPat();
        public Expr ZeroPoint() => GetCast<Expr>(ZeroPointPat());
        public T ZeroPoint<T>()
            where T : Expr => GetCast<T>(ZeroPointPat());
        public ExprPattern ScalePat() => Pattern[Quantize.Scale];
        public T ScalePat<T>()
            where T : ExprPattern => (T)ScalePat();
        public Expr Scale() => GetCast<Expr>(ScalePat());
        public T Scale<T>()
            where T : Expr => GetCast<T>(ScalePat());
        public DataType TargetType => ((Quantize)GetCast<Call>(this).Target).TargetType;
        public static implicit operator CallPattern(QuantizeWrapper warper) => warper.Pattern;
    }

    public sealed record ReduceWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Reduce.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern AxisPat() => Pattern[Reduce.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public ExprPattern InitValuePat() => Pattern[Reduce.InitValue];
        public T InitValuePat<T>()
            where T : ExprPattern => (T)InitValuePat();
        public Expr InitValue() => GetCast<Expr>(InitValuePat());
        public T InitValue<T>()
            where T : Expr => GetCast<T>(InitValuePat());
        public ExprPattern KeepDimsPat() => Pattern[Reduce.KeepDims];
        public T KeepDimsPat<T>()
            where T : ExprPattern => (T)KeepDimsPat();
        public Expr KeepDims() => GetCast<Expr>(KeepDimsPat());
        public T KeepDims<T>()
            where T : Expr => GetCast<T>(KeepDimsPat());
        public ReduceOp ReduceOp => ((Reduce)GetCast<Call>(this).Target).ReduceOp;
        public static implicit operator CallPattern(ReduceWrapper warper) => warper.Pattern;
    }

    public sealed record ReduceWindow2DWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[ReduceWindow2D.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern InitValuePat() => Pattern[ReduceWindow2D.InitValue];
        public T InitValuePat<T>()
            where T : ExprPattern => (T)InitValuePat();
        public Expr InitValue() => GetCast<Expr>(InitValuePat());
        public T InitValue<T>()
            where T : Expr => GetCast<T>(InitValuePat());
        public ExprPattern FilterPat() => Pattern[ReduceWindow2D.Filter];
        public T FilterPat<T>()
            where T : ExprPattern => (T)FilterPat();
        public Expr Filter() => GetCast<Expr>(FilterPat());
        public T Filter<T>()
            where T : Expr => GetCast<T>(FilterPat());
        public ExprPattern StridePat() => Pattern[ReduceWindow2D.Stride];
        public T StridePat<T>()
            where T : ExprPattern => (T)StridePat();
        public Expr Stride() => GetCast<Expr>(StridePat());
        public T Stride<T>()
            where T : Expr => GetCast<T>(StridePat());
        public ExprPattern PaddingPat() => Pattern[ReduceWindow2D.Padding];
        public T PaddingPat<T>()
            where T : ExprPattern => (T)PaddingPat();
        public Expr Padding() => GetCast<Expr>(PaddingPat());
        public T Padding<T>()
            where T : Expr => GetCast<T>(PaddingPat());
        public ExprPattern DilationPat() => Pattern[ReduceWindow2D.Dilation];
        public T DilationPat<T>()
            where T : ExprPattern => (T)DilationPat();
        public Expr Dilation() => GetCast<Expr>(DilationPat());
        public T Dilation<T>()
            where T : Expr => GetCast<T>(DilationPat());
        public ReduceOp ReduceOp => ((ReduceWindow2D)GetCast<Call>(this).Target).ReduceOp;
        public static implicit operator CallPattern(ReduceWindow2DWrapper warper) => warper.Pattern;
    }

    public sealed record ReshapeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Reshape.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern ShapePat() => Pattern[Reshape.Shape];
        public T ShapePat<T>()
            where T : ExprPattern => (T)ShapePat();
        public Expr Shape() => GetCast<Expr>(ShapePat());
        public T Shape<T>()
            where T : Expr => GetCast<T>(ShapePat());
        public static implicit operator CallPattern(ReshapeWrapper warper) => warper.Pattern;
    }

    public sealed record ResizeImageWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[ResizeImage.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern NewSizePat() => Pattern[ResizeImage.NewSize];
        public T NewSizePat<T>()
            where T : ExprPattern => (T)NewSizePat();
        public Expr NewSize() => GetCast<Expr>(NewSizePat());
        public T NewSize<T>()
            where T : Expr => GetCast<T>(NewSizePat());
        public ExprPattern AlignCornersPat() => Pattern[ResizeImage.AlignCorners];
        public T AlignCornersPat<T>()
            where T : ExprPattern => (T)AlignCornersPat();
        public Expr AlignCorners() => GetCast<Expr>(AlignCornersPat());
        public T AlignCorners<T>()
            where T : Expr => GetCast<T>(AlignCornersPat());
        public ExprPattern HalfPixelCentersPat() => Pattern[ResizeImage.HalfPixelCenters];
        public T HalfPixelCentersPat<T>()
            where T : ExprPattern => (T)HalfPixelCentersPat();
        public Expr HalfPixelCenters() => GetCast<Expr>(HalfPixelCentersPat());
        public T HalfPixelCenters<T>()
            where T : Expr => GetCast<T>(HalfPixelCentersPat());
        public ImageResizeMode ResizeMode => ((ResizeImage)GetCast<Call>(this).Target).ResizeMode;
        public static implicit operator CallPattern(ResizeImageWrapper warper) => warper.Pattern;
    }

    public sealed record ShapeOpWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[ShapeOp.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public static implicit operator CallPattern(ShapeOpWrapper warper) => warper.Pattern;
    }

    public sealed record SliceWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Slice.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern BeginsPat() => Pattern[Slice.Begins];
        public T BeginsPat<T>()
            where T : ExprPattern => (T)BeginsPat();
        public Expr Begins() => GetCast<Expr>(BeginsPat());
        public T Begins<T>()
            where T : Expr => GetCast<T>(BeginsPat());
        public ExprPattern EndsPat() => Pattern[Slice.Ends];
        public T EndsPat<T>()
            where T : ExprPattern => (T)EndsPat();
        public Expr Ends() => GetCast<Expr>(EndsPat());
        public T Ends<T>()
            where T : Expr => GetCast<T>(EndsPat());
        public ExprPattern AxesPat() => Pattern[Slice.Axes];
        public T AxesPat<T>()
            where T : ExprPattern => (T)AxesPat();
        public Expr Axes() => GetCast<Expr>(AxesPat());
        public T Axes<T>()
            where T : Expr => GetCast<T>(AxesPat());
        public ExprPattern StridesPat() => Pattern[Slice.Strides];
        public T StridesPat<T>()
            where T : ExprPattern => (T)StridesPat();
        public Expr Strides() => GetCast<Expr>(StridesPat());
        public T Strides<T>()
            where T : Expr => GetCast<T>(StridesPat());
        public static implicit operator CallPattern(SliceWrapper warper) => warper.Pattern;
    }

    public sealed record SpaceToBatchWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[SpaceToBatch.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern BlockShapePat() => Pattern[SpaceToBatch.BlockShape];
        public T BlockShapePat<T>()
            where T : ExprPattern => (T)BlockShapePat();
        public Expr BlockShape() => GetCast<Expr>(BlockShapePat());
        public T BlockShape<T>()
            where T : Expr => GetCast<T>(BlockShapePat());
        public ExprPattern PaddingsPat() => Pattern[SpaceToBatch.Paddings];
        public T PaddingsPat<T>()
            where T : ExprPattern => (T)PaddingsPat();
        public Expr Paddings() => GetCast<Expr>(PaddingsPat());
        public T Paddings<T>()
            where T : Expr => GetCast<T>(PaddingsPat());
        public static implicit operator CallPattern(SpaceToBatchWrapper warper) => warper.Pattern;
    }

    public sealed record SplitWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Split.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern AxisPat() => Pattern[Split.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public ExprPattern SectionsPat() => Pattern[Split.Sections];
        public T SectionsPat<T>()
            where T : ExprPattern => (T)SectionsPat();
        public Expr Sections() => GetCast<Expr>(SectionsPat());
        public T Sections<T>()
            where T : Expr => GetCast<T>(SectionsPat());
        public static implicit operator CallPattern(SplitWrapper warper) => warper.Pattern;
    }

    public sealed record SqueezeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Squeeze.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern DimPat() => Pattern[Squeeze.Dim];
        public T DimPat<T>()
            where T : ExprPattern => (T)DimPat();
        public Expr Dim() => GetCast<Expr>(DimPat());
        public T Dim<T>()
            where T : Expr => GetCast<T>(DimPat());
        public static implicit operator CallPattern(SqueezeWrapper warper) => warper.Pattern;
    }

    public sealed record StackWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputsPat() => Pattern[Stack.Inputs];
        public T InputsPat<T>()
            where T : ExprPattern => (T)InputsPat();
        public Expr Inputs() => GetCast<Expr>(InputsPat());
        public T Inputs<T>()
            where T : Expr => GetCast<T>(InputsPat());
        public ExprPattern AxisPat() => Pattern[Stack.Axis];
        public T AxisPat<T>()
            where T : ExprPattern => (T)AxisPat();
        public Expr Axis() => GetCast<Expr>(AxisPat());
        public T Axis<T>()
            where T : Expr => GetCast<T>(AxisPat());
        public static implicit operator CallPattern(StackWrapper warper) => warper.Pattern;
    }

    public sealed record TransposeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[Transpose.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern PermPat() => Pattern[Transpose.Perm];
        public T PermPat<T>()
            where T : ExprPattern => (T)PermPat();
        public Expr Perm() => GetCast<Expr>(PermPat());
        public T Perm<T>()
            where T : Expr => GetCast<T>(PermPat());
        public static implicit operator CallPattern(TransposeWrapper warper) => warper.Pattern;
    }

    public sealed record UnSqueezeWrapper(CallPattern Pattern) : PatternWrapper
    {
        public ExprPattern InputPat() => Pattern[UnSqueeze.Input];
        public T InputPat<T>()
            where T : ExprPattern => (T)InputPat();
        public Expr Input() => GetCast<Expr>(InputPat());
        public T Input<T>()
            where T : Expr => GetCast<T>(InputPat());
        public ExprPattern DimPat() => Pattern[UnSqueeze.Dim];
        public T DimPat<T>()
            where T : ExprPattern => (T)DimPat();
        public Expr Dim() => GetCast<Expr>(DimPat());
        public T Dim<T>()
            where T : Expr => GetCast<T>(DimPat());
        public static implicit operator CallPattern(UnSqueezeWrapper warper) => warper.Pattern;
    }
}