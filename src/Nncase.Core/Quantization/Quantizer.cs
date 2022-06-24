// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.Quantization;

public class Quantizer
{
    private readonly Function _function;
    private readonly List<Call> _rangeOfs = new List<Call>();

    public Quantizer(Function function)
    {
        _function = function;
        FillRangeOfs();
    }

    public void Step(IReadOnlyDictionary<Var, IValue> inputs)
    {

    }

    private void FillRangeOfs()
    {
    }

    private class RangeOfVisitor : ExprVisitor<Expr, IRType>
    {
        public override Expr VisitLeaf(Call expr)
        {
            if (expr.Target is Op op
                && op is RangeOf)
            {

            }

            return base.VisitLeaf(expr);
        }

        public override Expr DefaultVisitLeaf(Expr expr)
        {
            return expr;
        }
    }
}
