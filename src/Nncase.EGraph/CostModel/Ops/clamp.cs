// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR.Math;
using Nncase.IR;
using System.Linq;
using System;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        /// <summary>
        /// Compute clamp cost ，size * 2.
        /// </summary>
        /// <param name="binary"></param>
        /// <returns></returns>
        private Cost VisitClamp(Clamp clamp)
        {
            var shape = _context.GetArgumentType(clamp, Clamp.Input) as TensorType;
            var arithm = (shape.Shape.Prod() * 2).FixedValue;
            return new Cost(arithm);
        }
    }
}