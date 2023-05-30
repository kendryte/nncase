// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Reactive;
using Nncase.IR;

namespace Nncase
{
    public class FunctionCollector : ExprVisitor<int, IRType>
    {
        private readonly HashSet<Function> _functions = new(ReferenceEqualityComparer.Instance);

        public FunctionCollector()
            : base(true)
        {
        }

        public HashSet<Function> Functions => _functions;

        protected override int VisitLeafFunction(Function expr, Unit context)
        {
            _functions.Add(expr);
            return 0;
        }

        protected override int DefaultVisitLeaf(Expr expr) => 1;
    }
}
