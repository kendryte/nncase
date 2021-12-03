// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitClip(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var min = GetFloatAttribute(op, "min", float.MinValue);
            var max = GetFloatAttribute(op, "max", float.MaxValue);
            return F.Math.Clamp(input, min, max);
        }
    }
}