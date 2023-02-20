// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase
{
    public class FunctionCollector : ExprVisitor<int, IRType>
    {
        private readonly HashSet<Function> _functions = new(ReferenceEqualityComparer.Instance);

        public HashSet<Function> Functions => _functions;

        public override int VisitLeaf(Function expr)
        {
            _functions.Add(expr);
            return 0;
        }

        public override int DefaultVisitLeaf(Expr expr) => 1;
    }
}
