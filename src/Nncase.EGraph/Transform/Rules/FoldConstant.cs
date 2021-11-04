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

namespace Nncase.Transform.Rule
{
    public class FoldConstantBase : EGraphRule
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