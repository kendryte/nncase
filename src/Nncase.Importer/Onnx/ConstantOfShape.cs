// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using static Onnx.AttributeProto.Types;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConstantOfShape(in NodeProto op)
        {
            var shape = GetInputExpr(op, 0);
            var tensorValue = GetAttr(op, "value", AttributeType.Tensor, x => x.T);
            if (tensorValue)
            {
                var tensor = tensorValue.ValueUnsafe();
                var tensorConst = GetTensor(tensor);
                var type = GetDataType(tensor);
                return ConstantOfShape(shape, GetTensor(tensor));
            }
            else
            {
                return F.Tensors.ConstantOfShape(shape, Tensor.From<float>(new[] { 0f }));
            }
        }
    }
}
