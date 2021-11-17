using System.Linq;
using System.Collections.Immutable;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Math;
using Nncase.Pattern;
using Nncase.IR.Tensors;
using Nncase.IR.Math;
using Nncase.IR;

namespace Nncase.Transform.Rule
{
    public class FoldConstantBase : PatternRule
    {
        public List<ConstPattern> wcconsts = new();

        public VArgsPattern wcargs;

        public FoldConstantBase()
        {
            wcargs = IsVArgsRepeat(
                     (n, param) =>
                     {
                         for (int i = 0; i < n; i++)
                         {
                             var wcconst = IsConst();
                             param.Add(wcconst);
                             wcconsts.Add(wcconst);
                         }
                     },
                     (match, param) =>
                     {
                         if (!match)
                         {
                             param.Clear();
                             wcconsts.Clear();
                         }
                     }
                   );
        }
    }

    public class FoldConstantCall : FoldConstantBase
    {
        public WildCardPattern wctarget = "target";
        public CallPattern wccall;
        public FoldConstantCall()
        {
            wccall = IsCall(wctarget, wcargs);
            Pattern = wccall;
        }
    }

    public class FoldConstantFunction : FoldConstantBase
    {
        public WildCardPattern wcbody = "body";
        public FunctionPattern wcfunc;
        public FoldConstantFunction()
        {
            wcfunc = IsFunction(wcbody, wcargs);
        }
    }
}