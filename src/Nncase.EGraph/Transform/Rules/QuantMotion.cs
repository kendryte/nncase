// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Pattern;
using Nncase.Pattern.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Imaging;

namespace Nncase.Transform.Rule
{
    public class QuantPadMotion : PatternRule
    {
        QuantizeWrapper quant;
        PadWrapper pad;
        public QuantPadMotion()
        {
            pad = Pad(IsWildCard(), IsConst(), PadMode.Constant, IsConst());
            Pattern = quant = IsQuantize(pad);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            pad.Bind(result);
            quant.Bind(result);

            var old_padv = pad.Value<TensorConst>().Value.ToScalar<float>();
            var zero_point = quant.ZeroPoint<TensorConst>().Value.ToScalar<int>();
            var scale = quant.Scale<TensorConst>().Value.ToScalar<float>();
            var new_padv = Math.Clamp((int)Math.Round((old_padv - zero_point) * scale), 0, 255);
            return Pad(Quantize(pad.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), pad.Pads(), pad.PadMode, new_padv);
        }
    }

    public class QuantTransposeMotion : PatternRule
    {
        private TransposeWrapper trans;
        private QuantizeWrapper quant;

        public QuantTransposeMotion()
        {
            trans = Transpose(IsWildCard(), IsConstIntTensor());
            Pattern = quant = IsQuantize(trans);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            trans.Bind(result);
            quant.Bind(result);
            return Transpose(Quantize(trans.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), trans.Perm());
        }
    }

    public class QuantSliceMotion : PatternRule
    {
        private SliceWrapper slice;
        private QuantizeWrapper quant;

        public QuantSliceMotion()
        {
            slice = IsSlice(IsWildCard());
            Pattern = quant = IsQuantize(slice);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            slice.Bind(result);
            quant.Bind(result);

            return Slice(Quantize(slice.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), slice.Begins(), slice.Ends(), slice.Axes(), slice.Strides());
        }
    }

    public class QuantResizeMotion : PatternRule
    {
        private ResizeImageWrapper resize;
        private QuantizeWrapper quant;

        public QuantResizeMotion()
        {
            resize = IsResize(IsWildCard(), IsWildCard());
            Pattern = quant = IsQuantize(resize);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            resize.Bind(result);
            quant.Bind(result);
            return ResizeImage(resize.ResizeMode, Quantize(resize.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), resize.NewSize(), resize.AlignCorners(), resize.HalfPixelCenters());
        }
    }

    public class QuantReshapeMotion : PatternRule
    {
        private ReshapeWrapper reshape;
        private QuantizeWrapper quant;

        public QuantReshapeMotion()
        {
            reshape = Reshape(IsWildCard(), IsWildCard());
            Pattern = quant = IsQuantize(reshape);
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            reshape.Bind(result);
            quant.Bind(result);
            return Reshape(Quantize(reshape.Input(), quant.ZeroPoint(), quant.Scale(), quant.TargetType), reshape.Shape());
        }
    }

    public class QuantBatchToSpaceMotion : PatternRule
    {
        private QuantizeWrapper quant;
        private BatchToSpaceWrapper b2s;

        public QuantBatchToSpaceMotion()
        {
            quant = IsQuantize(IsWildCard());
            Pattern = b2s = BatchToSpace(quant, IsWildCard(), IsWildCard());
        }

        public override Expr? GetRePlace(IMatchResult result)
        {
            quant.Bind(result);
            b2s.Bind(result);
            return BatchToSpace(quant.Input(), b2s.BlockShape(), b2s.Crops());
        }
    }
}