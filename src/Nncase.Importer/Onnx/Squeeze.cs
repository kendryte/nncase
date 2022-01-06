// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitSqueeze(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axes = GetOptionInputExpr(op, 1, SqueezeAxes(input));
            return Squeeze(input, axes);
        }

        // todo:default is error
        private Expr SqueezeAxes(Expr input)
        {
            return ReduceArg(ReduceArgOp.ArgMin, ShapeOp(input), 0, true, false);
        }
    }
}