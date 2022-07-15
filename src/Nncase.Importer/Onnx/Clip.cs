// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitClip(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var dt = GetInputDataType(op, 0);
            var min = GetOptionInputExpr(op, 1).Match(min => Squeeze(min, new[]{0}), Cast(float.MinValue, dt));
            var max = GetOptionInputExpr(op, 2).Match(max => Squeeze(max, new[]{0}), Cast(float.MaxValue, dt));
            return F.Math.Clamp(input, min, max);
        }
    }
}