// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using LanguageExt;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Onnx;
using static Onnx.AttributeProto.Types;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConstant(in NodeProto op)
        {
            var tensorValue = GetAttr(op, "value", AttributeType.Tensor, x => x.T);
            if (tensorValue)
            {
                return GetTensor(tensorValue.ValueUnsafe());
            }

            var floatValue = GetAttr(op, "value_float", AttributeType.Float, x => x.F);
            if (floatValue)
            {
                return Tensor.FromScalar(floatValue.Value());
            }

            var floatsValue = GetAttr(op, "value_floats", AttributeType.Floats, x => x.Floats);
            if (floatsValue)
            {
                var floats = floatsValue.ValueUnsafe();
                return Tensor.From<float>(floats.ToArray(), new Shape(floats.Count));
            }

            var intValue = GetAttr(op, "value_int", AttributeType.Int, x => x.I);

            if (intValue)
            {
                return Tensor.FromScalar(intValue.Value());
            }

            var intsValue = GetAttr(op, "value_ints", AttributeType.Ints, x => x.Ints);

            if (intsValue)
            {
                var ints = intsValue.ValueUnsafe();
                return Tensor.From<long>(ints.ToArray(), new Shape(ints.Count));
            }

            throw new NotSupportedException("Constant field format is not supported.");
        }
    }
}
