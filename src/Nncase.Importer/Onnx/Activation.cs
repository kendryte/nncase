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
            return SetOutputsNames(Elu(input, alpha), op);
        }

        private Expr VisitCelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 1.0f);
            return SetOutputsNames(Celu(input, alpha), op);
        }

        private Expr VisitRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return SetOutputsNames(Relu(input), op);
        }

        private Expr VisitLeakyRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 0.01f);
            return SetOutputsNames(LeakyRelu(input, alpha), op);
        }

        private Expr VisitPRelu(NodeProto op)
        {
            var (input, slope) = GetInputExprs(op, 0, 1);
            return SetOutputsNames(PRelu(input, slope), op);
        }

        private Expr VisitSelu(NodeProto op)
        {
            var x = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 1.67326319217681884765625F);
            var gamma = GetFloatAttribute(op, "gamma", 1.05070102214813232421875F);
            return SetOutputsNames(Selu(x, alpha, gamma), op);
        }

        private Expr VisitSigmoid(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return SetOutputsNames(Sigmoid(input), op);
        }

        private Expr VisitHardSigmoid(NodeProto op)
        {
            var x = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 0.2f);
            var beta = GetFloatAttribute(op, "beta", 0.5f);
            return SetOutputsNames(HardSigmoid(x, alpha, beta), op);
        }

        private Expr VisitHardSwish(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return SetOutputsNames(HardSwish(input), op);
        }

        private Expr VisitErf(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return SetOutputsNames(Erf(input), op);
        }
    }
}
