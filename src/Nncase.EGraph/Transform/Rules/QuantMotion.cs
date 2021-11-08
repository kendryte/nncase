using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern;
using Nncase.Transform.Pattern.Tensors;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;

namespace Nncase.Transform.Rule
{
    public class QuantPadMotion : EGraphRule
    {
        QuantizeWrapper quant;
        PadWrapper pad;
        public QuantPadMotion()
        {
            pad = Pad(IsWildCard(), IsConst(), PadMode.Constant, IsConst());
            Pattern = quant = IsQuantize(pad);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            pad.Bind(result);
            quant.Bind(result);

            var old_padv = pad.Value<Const>().ToScalar<float>();
            var zero_point = quant.ZeroPoint<Const>().ToScalar<int>();
            var scale = quant.Scale<Const>().ToScalar<float>();
            var new_padv = Math.Clamp((int)Math.Round((old_padv - zero_point) * scale), 0, 255);
            return Pad(Quantize(pad.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), pad.Pads(), pad.PadMode, new_padv);
        }
    }

    public class QuantTransposeMotion : EGraphRule
    {
        private TransposeWrapper trans;
        private QuantizeWrapper quant;

        public QuantTransposeMotion()
        {
            trans = Transpose(IsWildCard(), IsConstIntTensor());
            Pattern = quant = IsQuantize(trans);
        }
        public override Expr? GetRePlace(EMatchResult result)
        {
            trans.Bind(result);
            quant.Bind(result);
            return Transpose(Quantize(trans.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), trans.Perm());
        }
    }

    public class QuantSliceMotion : EGraphRule
    {
        private SliceWrapper slice;
        private QuantizeWrapper quant;

        public QuantSliceMotion()
        {
            slice = IsSlice(IsWildCard());
            Pattern = quant = IsQuantize(slice);
        }
        public override Expr? GetRePlace(EMatchResult result)
        {
            slice.Bind(result);
            quant.Bind(result);

            return Slice(Quantize(slice.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), slice.Begins(), slice.Ends(), slice.Axes(), slice.Strides());
        }
    }

    public class QuantResizeMotion : EGraphRule
    {
        private ResizeImageWrapper resize;
        private QuantizeWrapper quant;

        public QuantResizeMotion()
        {
            resize = IsResize(IsWildCard(), IsWildCard());
            Pattern = quant = IsQuantize(resize);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            resize.Bind(result);
            quant.Bind(result);
            return ResizeImage(resize.ResizeMode, Quantize(resize.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), resize.NewSize(), resize.AlignCorners(), resize.HalfPixelCenters());
        }
    }

    public class QuantReshapeMotion : EGraphRule
    {
        private ReshapeWrapper reshape;
        private QuantizeWrapper quant;

        public QuantReshapeMotion()
        {
            reshape = Reshape(IsWildCard(), IsWildCard());
            Pattern = quant = IsQuantize(reshape);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            reshape.Bind(result);
            quant.Bind(result);
            return Reshape(Quantize(reshape.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), reshape.Shape());
        }
    }

    public class QuantBatchToSpaceMotion : EGraphRule
    {
        private QuantizeWrapper quant;
        private BatchToSpaceWrapper b2s;

        public QuantBatchToSpaceMotion()
        {
            quant = IsQuantize(IsWildCard());
            Pattern = b2s = BatchToSpace(quant, IsWildCard(), IsWildCard());
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            quant.Bind(result);
            b2s.Bind(result);
            return BatchToSpace(quant.Input(), b2s.BlockShape(), b2s.Crops());
        }
    }
}