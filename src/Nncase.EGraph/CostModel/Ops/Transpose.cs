// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using System.Linq;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        private Cost VisitTranspose(Transpose transpose)
        {
            var type = _context.CurrentCallResultTensorType();
            var perm = _context.GetArgumentValue(transpose, Transpose.Perm).AsTensor().Cast<int>();
            int arithm = -1;
            foreach (var (i, p) in Enumerable.Range(0, (int)perm.Length).Zip(perm))
            {
                if (arithm != -1)
                {
                    break;
                }

                if (i == p)
                {
                    continue;
                }

                // find axis which transpose start.
                // the inner axis have more weights.
                foreach (var j in Enumerable.Range(i + 1, (int)perm.Length))
                {
                    if (perm[j] == i)
                    {
                        arithm = type.Shape.Skip(i).Aggregate(new Dimension(1), (x, y) => x * y).FixedValue * ((int)perm.Length - i);
                        break;
                    }
                }
            }

            return new(arithm == -1 ? 1 : arithm, type.DType.SizeInBytes);
        }
    }
}