// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitCelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 1.0f);
            return F.NN.Celu(input, alpha);
        }
        
        private Expr VisitRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return F.NN.Relu(input);
        }
        
        private Expr VisitLeakyRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return F.NN.LeakyRelu(input);
        }
        
        private Expr VisitPRelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return F.NN.PRelu(input);
        }
        
        private Expr VisitSelu(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            return F.NN.Selu(input);
        }
    }
}