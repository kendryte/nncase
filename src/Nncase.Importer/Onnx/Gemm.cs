﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitGemm(in NodeProto op)
        {
            var (a, b) = GetInputExprs(op, 0, 1);
            var alpha = GetFloatAttribute(op, "alpha", 1.0f);
            var transA = GetBoolAttribute(op, "transA", false);
            var transB = GetBoolAttribute(op, "transB", false);
            if (transA)
            {
                a = F.Tensors.Transpose(a, new[] { 1, 0 });
            }

            if (transB)
            {
                b = F.Tensors.Transpose(b, new[] { 1, 0 });
            }

            var gemm = F.Tensors.MatMul(a, b) * alpha;
            if (op.Input.Count == 3)
            {
                var c = GetInputExpr(op, 2);
                var beta = GetFloatAttribute(op, "beta", 1.0f);
                return gemm + (beta * c);
            }
            else
            {
                return gemm;
            }
        }
    }
}
