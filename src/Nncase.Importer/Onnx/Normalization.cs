// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitBatchNormalization(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var eps = GetFloatAttribute(op, "epsilon", 1e-05f);
            var mom = GetFloatAttribute(op, "momentum", 0.9f);
            return F.NN.BatchNormalization(input, eps, mom);
        }

        private Expr VisitInstanceNormalization(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var (scale, bias) = GetInputExprs(op, 1, 2);
            var eps = GetFloatAttribute(op, "epsilon", 1e-05f);
            return F.NN.InstanceNormalization(input, eps) * scale + bias;
        }

        private Expr VisitLpNormalization(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axis = GetIntAttribute(op, "axis", -1);
            var p = GetIntAttribute(op, "p", 2);
            return F.NN.LpNormalization(input, axis, p);
        }
    }
}