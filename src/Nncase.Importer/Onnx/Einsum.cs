// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitEinsum(in NodeProto op)
        {
            // TODO: only support two inputs and '->' can not be ommitted
            var equation = GetStringAttribute(op, "equation", string.Empty);
            if (string.IsNullOrEmpty(equation) || equation.Count(c => c == ',') != 1)
            {
                throw new InvalidOperationException("Not Yet Supported Einsum Operation!");
            }

            var inTerm1 = equation.Split(',')[0];
            var remains = equation.Split(',')[1];
            var inTerm2 = remains.Substring(0, remains.IndexOf('-', System.StringComparison.Ordinal));
            var outTerm = remains.Split('>')[1];
            var (lhs, rhs) = GetInputExprs(op, 0, 1);

            // i,j->ij
            if (inTerm1.Length == 1 && inTerm2.Length == 1 && outTerm.Length == 2 && inTerm1 + inTerm2 == outTerm)
            {
                return F.Tensors.Unsqueeze(lhs, new[] { 1 }) * F.Tensors.Unsqueeze(rhs, new[] { 0 });
            }

            // ibh,hnd->ibnd
            if (inTerm1.Length == 3 && inTerm2.Length == 3 && outTerm.Length == 4
            && inTerm1.Substring(0, 2) + inTerm2.Substring(1, 2) == outTerm
            && inTerm1.Last() == inTerm2.First())
            {
                var lhsShape = F.Tensors.ShapeOf(lhs);
                var rhsShape = F.Tensors.ShapeOf(rhs);
                var mm = F.Math.MatMul(lhs, F.Tensors.Reshape(rhs, F.Tensors.Stack(new IR.Tuple(rhsShape[0], rhsShape[1] * rhsShape[2]), 0)));
                return F.Tensors.Reshape(mm, F.Tensors.Stack(new IR.Tuple(lhsShape[0], lhsShape[1], rhsShape[1], rhsShape[2]), 0));
            }

            // ibnd,jbnd->bnij
            if (inTerm1.Length == 4 && inTerm2.Length == 4 && outTerm.Length == 4
            && inTerm1.Substring(1, 2) + inTerm1.First() + inTerm2.First() == outTerm
            && inTerm1.Substring(1, 2) == inTerm2.Substring(1, 2)
            && inTerm1.Last() == inTerm2.Last())
            {
                var lhsShape = F.Tensors.ShapeOf(lhs);
                var rhsShape = F.Tensors.ShapeOf(rhs);
                var mm = F.Math.MatMul(
                F.Tensors.Transpose(F.Tensors.Reshape(lhs, F.Tensors.Stack(new IR.Tuple(lhsShape[0], lhsShape[1] * lhsShape[2], lhsShape[3]), 0)), new[] { 1, 0, 2 }),
                F.Tensors.Transpose(F.Tensors.Reshape(rhs, F.Tensors.Stack(new IR.Tuple(rhsShape[0], rhsShape[1] * rhsShape[2], rhsShape[3]), 0)), new[] { 1, 2, 0 }));

                return F.Tensors.Reshape(mm, F.Tensors.Stack(new IR.Tuple(lhsShape[1], lhsShape[2], lhsShape[0], rhsShape[0]), 0));
            }

            // bnij,jbnd->ibnd
            if (inTerm1.Length == 4 && inTerm2.Length == 4 && outTerm.Length == 4
                && inTerm1[2] + inTerm1.Substring(0, 2) + inTerm2.Last() == outTerm
                && inTerm1.Substring(0, 2) == inTerm2.Substring(1, 2)
                && inTerm1.Last() == inTerm2.First())
            {
                var lhsShape = F.Tensors.ShapeOf(lhs);
                var rhsShape = F.Tensors.ShapeOf(rhs);
                var mm = F.Math.MatMul(
                F.Tensors.Reshape(lhs, F.Tensors.Stack(new IR.Tuple(lhsShape[0] * lhsShape[1], lhsShape[2], lhsShape[3]), 0)),
                F.Tensors.Transpose(F.Tensors.Reshape(rhs, F.Tensors.Stack(new IR.Tuple(rhsShape[0], rhsShape[1] * rhsShape[2], rhsShape[3]), 0)), new[] { 1, 0, 2 }));

                return F.Tensors.Reshape(F.Tensors.Transpose(mm, new[] { 1, 0, 2 }), F.Tensors.Stack(new IR.Tuple(lhsShape[2], lhsShape[0], lhsShape[1], rhsShape[3]), 0));
            }

            // ibnd,hnd->ibh
            if (inTerm1.Length == 4 && inTerm2.Length == 3 && outTerm.Length == 3
                && inTerm1.Substring(0, 2) + inTerm2.First() == outTerm
                && inTerm1.Substring(2, 2) == inTerm2.Substring(1, 2))
            {
                var lhsShape = F.Tensors.ShapeOf(lhs);
                var rhsShape = F.Tensors.ShapeOf(rhs);
                var mm = F.Math.MatMul(
                F.Tensors.Reshape(lhs, F.Tensors.Stack(new IR.Tuple(lhsShape[0], lhsShape[1], lhsShape[2] * lhsShape[3]), 0)),
                F.Tensors.Transpose(F.Tensors.Reshape(rhs, F.Tensors.Stack(new IR.Tuple(rhsShape[0], rhsShape[1] * rhsShape[2]), 0)), new[] { 1, 0 }));

                return F.Tensors.Reshape(mm, F.Tensors.Stack(new IR.Tuple(lhsShape[0], lhsShape[1], rhsShape[0]), 0));
            }

            throw new InvalidOperationException("Not Yet Supported Einsum Operation!");
        }
    }
}
