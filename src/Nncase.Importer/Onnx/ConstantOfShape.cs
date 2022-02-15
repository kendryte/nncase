// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;
using static Onnx.AttributeProto.Types;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConstantOfShape(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var shape = new Shape();
            var tensorValue = GetAttr(op, "value", AttributeType.Tensor, x => x.T);
            if (tensorValue)
            {
                var tensor = tensorValue.ValueUnsafe();
                var tensorConst = GetTensor(tensor);
                var type = GetDataType(tensor);
                if (type == DataTypes.Float32)
                {
                    return Tensor.FromScalar(tensorConst.ToScalar<float>(), shape);
                }
                else
                {
                    throw new NotSupportedException($"Not Supported type {type} in ConstantOfShape");
                }
            }
            else
            {
                return Tensor.FromScalar(0.0f, shape);
            }
        }
    }
}