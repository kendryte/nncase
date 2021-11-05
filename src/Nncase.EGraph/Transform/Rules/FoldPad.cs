using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Transform.Pattern;
using static Nncase.Transform.Pattern.F.Math;
using static Nncase.Transform.Pattern.F.Tensors;
using static Nncase.Transform.Pattern.Utility;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using System.Numerics.Tensors;


namespace Nncase.Transform.Rule
{
    public class FoldNopPad : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcpad = IsConst(IsTensor() & IsIntegral());
        public FoldNopPad()
        {
            Pattern = IsPad(wcin, wcpad, IsWildCard());
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var pad = result[wcpad].ToTensor<int>();
            if (pad.All(x => x == 0))
                return result[wcin];
            return null;
        }
    }

    public class FoldPadPad : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcpad1, wcpad2;
        ConstPattern wcvalue1, wcvalue2;
        CallPattern pad1, pad2;
        public FoldPadPad()
        {
            wcpad1 = IsConst(IsTensor() & IsIntegral());
            wcpad2 = wcpad1 with { };
            wcvalue1 = IsConst(IsScalar());
            wcvalue2 = wcvalue1 with { };
            pad1 = IsPad(wcin, wcpad1, wcvalue1);
            pad2 = IsPad(pad1, wcpad2, wcvalue2);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var (value1, value2) = result[wcvalue1, wcvalue2];
            var mode1 = ((Pad)result[pad1].Target).padMode;
            var mode2 = ((Pad)result[pad2].Target).padMode;
            if ((mode1 == mode2) && (mode1 != PadMode.Constant || value1.Data == value2.Data))
            {
                var (t1, t2) = (value1.ToTensor<int>(), value2.ToTensor<int>());
                var newt = new DenseTensor<int>(t1.Dimensions);
                for (int i = 0; i < t1.Dimensions[0]; i++)
                {
                    newt[i, 0] = t1[i, 0] + t2[i, 0];
                    newt[i, 1] = t1[i, 1] + t2[i, 1];
                }
                var newpad = Const.FromTensor<int>(newt);
                return Pad(result[wcin], newpad, mode1, value1);
            }
            return null;
        }
    }


    public class FoldPadStrideSlice : EGraphRule
    {
        WildCardPattern wcin = "input", wcvalue = "value";
        CallPattern wcpad;
        ConstPattern wcpads = IsConstIntTensor(), wcbegin = IsConstIntTensor(),
         wcend = IsConstIntTensor(), wcaxes = IsConstIntTensor(), wcstride = IsConstIntTensor();
        public FoldPadStrideSlice()
        {
            wcpad = IsPad(wcin, wcpads, wcvalue);
            Pattern = Slice(wcpad, wcbegin, wcend, wcaxes, wcstride);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var pads = result[wcpads].ToTensor<int>();
            if (pads.Any(x => x < 0))
            {
                var begin = result[wcbegin].ToTensor<int>();
                var end = result[wcend].ToTensor<int>();
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
                              ((Pad)result[wcpad].Target).padMode,
                              result[wcvalue]);
                    return Slice(newpad, newbegin, newend, result[wcaxes], result[wcstride]);
                }
            }
            return null;
        }
    }

    public class StrideSliceToPad : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcbegin = IsConstIntTensor(),
         wcend = IsConstIntTensor(), wcaxes = IsConstIntTensor(), wcstride = IsConst((int x) => x == 1);
        public StrideSliceToPad()
        {
            Pattern = Slice(wcin, wcbegin, wcend, wcaxes, wcstride);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var begin = result[wcbegin].ToTensor<int>();
            var end = result[wcend].ToTensor<int>();
            if (result[wcin].CheckedType is TensorType intype)
            {
                var paddings = new DenseTensor<int>(new[] { intype.Shape.Rank, 2 });
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

    public class PadToSlice : EGraphRule
    {
        WildCardPattern wcin = "input";
        ConstPattern wcpads = IsConst((int x) => x <= 0);
        WildCardPattern wcvalue = "value";
        public PadToSlice()
        {
            IsPad(wcin, wcpads, wcvalue);
        }

        public override Expr? GetRePlace(EMatchResult result)
        {
            var input = result[wcin];
            if (input.CheckedType is TensorType intype)
            {
                var shape = intype.Shape;
                var begin = new int[shape.Rank];
                var end = new int[shape.Rank];
                var padst = result[wcpads].ToTensor<int>();
                for (int i = 0; i < shape.Rank; i++)
                {
                    begin[i] = -padst[i, 0];
                    end[i] = padst[i, 1] + shape[i].FixedValue;
                }

                return Slice(result[wcin], Const.FromSpan<int>(begin), Const.FromSpan<int>(end));
            }
            return null;
        }
    }

    public class PadToMaxPool : EGraphRule
    {
      
    }
}