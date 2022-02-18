// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Numerics.Tensors;
using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.TypePatternUtility;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Math;
using Nncase.Pattern;
using Nncase.IR.Math;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Transform.Rule
{
    public class FoldNopPad : IRewriteRule
    {
        WildcardPattern wcin = "input";
        TensorConstPattern wcpad = IsTensorConst(IsIntegral());

        public FoldNopPad()
        {
            Pattern = IsPad(wcin, wcpad, IsWildcard());
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var pad = result[wcpad].Value.Cast<int>();
            if (pad.All(x => x == 0))
            {
                return result[wcin];
            }

            return null;
        }
    }

    public class FoldPadPad : IRewriteRule
    {
        WildcardPattern wcin = "input";
        TensorConstPattern wcpad1, wcpad2;
        TensorConstPattern wcvalue1, wcvalue2;
        CallPattern pad1, pad2;

        public FoldPadPad()
        {
            wcpad1 = IsTensorConst(IsIntegral());
            wcpad2 = wcpad1 with { };
            wcvalue1 = IsTensorConst(IsScalar());
            wcvalue2 = wcvalue1 with { };
            pad1 = IsPad(wcin, wcpad1, wcvalue1);
            pad2 = IsPad(pad1, wcpad2, wcvalue2);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var (value1, value2) = result[wcvalue1, wcvalue2];
            var mode1 = ((Pad)result[pad1].Target).PadMode;
            var mode2 = ((Pad)result[pad2].Target).PadMode;
            if ((mode1 == mode2) && (mode1 != PadMode.Constant || value1.Value.Equals(value2.Value)))
            {
                var (t1, t2) = (value1.Value.Cast<int>(), value2.Value.Cast<int>());
                var newt = new Tensor<int>(t1.Dimensions);
                for (int i = 0; i < t1.Dimensions[0]; i++)
                {
                    newt[i, 0] = t1[i, 0] + t2[i, 0];
                    newt[i, 1] = t1[i, 1] + t2[i, 1];
                }

                var newpad = Const.FromTensor(newt);
                return Pad(result[wcin], newpad, mode1, value1);
            }

            return null;
        }
    }

    public class FoldPadStrideSlice : IRewriteRule
    {
        WildcardPattern wcin = "input", wcvalue = "value";
        CallPattern wcpad;
        TensorConstPattern wcpads = IsConstIntTensor(), wcbegin = IsConstIntTensor(),
         wcend = IsConstIntTensor(), wcaxes = IsConstIntTensor(), wcstride = IsConstIntTensor();
        public FoldPadStrideSlice()
        {
            wcpad = IsPad(wcin, wcpads, wcvalue);
            Pattern = Slice(wcpad, wcbegin, wcend, wcaxes, wcstride);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var pads = result[wcpads].Value.Cast<int>();
            if (pads.Any(x => x < 0))
            {
                var begin = result[wcbegin].Value.Cast<int>();
                var end = result[wcend].Value.Cast<int>();
                if (begin.All(x => x >= 0) &&
                    end.All(x => x >= 0))
                {
                    for (int i = 0; i < pads.Rank; i++)
                    {
                        if (pads[i, 0] < 0)
                        {
                            var before = -pads[i, 0];
                            begin[i] += before;
                            end[i] += before;
                            pads[i, 0] = 0;
                        }

                        if (pads[i, 1] < 0)
                        {
                            pads[i, 1] = 0;
                        }
                    }

                    Const newbegin = Const.FromTensor(begin), newend = Const.FromTensor(end);
                    var newpad = Pad(result[wcin],
                              Const.FromTensor(pads),
                              ((Pad)result[wcpad].Target).PadMode,
                              result[wcvalue]);
                    return Slice(newpad, newbegin, newend, result[wcaxes], result[wcstride]);
                }
            }

            return null;
        }
    }

    public class StrideSliceToPad : IRewriteRule
    {
        WildcardPattern wcin = "input";
        TensorConstPattern wcbegin = IsConstIntTensor(),
         wcend = IsConstIntTensor(), wcaxes = IsConstIntTensor(), wcstride = IsConst((int x) => x == 1);
        public StrideSliceToPad()
        {
            Pattern = Slice(wcin, wcbegin, wcend, wcaxes, wcstride);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var begin = result[wcbegin].Value.Cast<int>();
            var end = result[wcend].Value.Cast<int>();
            if (result[wcin].CheckedType is TensorType intype)
            {
                var paddings = new Tensor<int>(new[] { intype.Shape.Rank, 2 });
                for (int i = 0; i < intype.Shape.Rank; i++)
                {
                    paddings[i, 0] = -begin[i];
                    paddings[i, 1] = end[i] - intype.Shape[i].FixedValue;
                }

                return Pad(result[wcin], Const.FromTensor(paddings), PadMode.Constant, (Const)(0));
            }

            return null;
        }
    }

    public class PadToSlice : IRewriteRule
    {
        WildcardPattern wcin = "input";
        TensorConstPattern wcpads = IsConst((int x) => x <= 0);
        WildcardPattern wcvalue = "value";
        public PadToSlice()
        {
            IsPad(wcin, wcpads, wcvalue);
        }

        public override Expr? GetReplace(IMatchResult result)
        {
            var input = result[wcin];
            if (input.CheckedType is TensorType intype)
            {
                var shape = intype.Shape;
                var begin = new int[shape.Rank];
                var end = new int[shape.Rank];
                var padst = result[wcpads].Value.Cast<int>();
                for (int i = 0; i < shape.Rank; i++)
                {
                    begin[i] = -padst[i, 0];
                    end[i] = padst[i, 1] + shape[i].FixedValue;
                }

                return Slice(result[wcin], Tensor.FromSpan<int>(begin), Tensor.FromSpan<int>(end), intype.Shape.Rank);
            }

            return null;
        }
    }
}