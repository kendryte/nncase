// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitElu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 1.0f);
            return Elu(input, alpha);
        }

        private Expr VisitCelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 1.0f);
            return Celu(input, alpha);
        }

        private Expr VisitRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return Relu(input);
        }

        private Expr VisitLeakyRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 0.01f);
            return LeakyRelu(input, alpha);
        }

        private Expr VisitPRelu(NodeProto op)
        {
            var (input, slope) = GetInputExprs(op, 0, 1);
            return PRelu(input, slope);
        }

        private Expr VisitSelu(NodeProto op)
        {
            var x = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 1.67326319217681884765625F);
            var gamma = GetFloatAttribute(op, "gamma", 1.05070102214813232421875F);
            return Selu(x, alpha, gamma);
        }

        private Expr VisitSigmoid(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return Sigmoid(input);
        }

        private Expr VisitHardSigmoid(NodeProto op)
        {
            var x = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 0.2f);
            var beta = GetFloatAttribute(op, "beta", 0.5f);
            return HardSigmoid(x, alpha, beta);
        }

        private Expr VisitHardSwish(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return HardSwish(input);
        }

        private Expr VisitErf(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return Erf(input);
        }
    }
}
