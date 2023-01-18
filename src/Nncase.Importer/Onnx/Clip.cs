// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using static Nncase.IR.F.Tensors;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitClip(in NodeProto op)
        {
            return GetOpSet(op) < 11 ? ClipV1(op) : ClipV11(op);
        }

        private Expr ClipV1(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var min = GetOptionFloatAttribute(op, "min").Or(float.MinValue);
            var max = GetOptionFloatAttribute(op, "max").Or(float.MaxValue);
            return F.Math.Clamp(input, min, max);
        }

        private Expr ClipV11(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var min = GetOptionInputExpr(op, 1).Match(
                min =>
                {
                    min.InferenceType();
                    return min.CheckedShape.IsScalar ? min : Squeeze(min, new[] { 0 });
                },
                Cast(float.MinValue, DataTypes.Float32));
            var max = GetOptionInputExpr(op, 2).Match(
                max =>
                    {
                        max.InferenceType();
                        return max.CheckedShape.IsScalar ? max : Squeeze(max, new[] { 0 });
                    },
                Cast(float.MaxValue, DataTypes.Float32));
            return F.Math.Clamp(input, min, max);
        }
    }
}
