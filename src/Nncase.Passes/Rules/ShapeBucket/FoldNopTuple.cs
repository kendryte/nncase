// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;

public class FoldNopTuple : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        int i = 0;
        while (true)
        {
            var preHash = input.GetHashCode();
            DumpScope.Current.DumpIR(input, $"{i}_before");

            IRHelpers.DCE(input);
            new FoldNopTupleVisitior().Visit(input);
            DumpScope.Current.DumpIR(input, $"{i++}_after_convert");
            var afterHash = input.GetHashCode();
            if (preHash == afterHash)
            {
                return Task.FromResult(input);
            }
        }
    }

    internal class FoldNopTupleVisitior : ExprVisitor<Expr, Unit>
    {
        private bool _changed;

        public FoldNopTupleVisitior()
            : base(true)
        {
        }

        protected override Expr DefaultVisitLeaf(Expr expr) => expr;

        protected override Expr VisitLeafTuple(Tuple expr)
        {
            if (!_changed && expr.Users.All(user => user is Call { Target: GetItem }))
            {
                foreach (var user in expr.Users)
                {
                    var index = ((TensorConst)((Call)user).Arguments[GetItem.Index.Index]).Value.ToScalar<int>();
                    ReplaceUtility.ReplaceAllUsesWith(user, expr.Fields[index]);
                }

                _changed = true;
            }

            return expr;
        }
    }
}
