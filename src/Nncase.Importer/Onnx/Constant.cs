// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using LanguageExt;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;
using static Onnx.AttributeProto.Types;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitConstant(in NodeProto op)
        {
            var tensorValue = GetAttr(op, "value", AttributeType.Tensor, x => x.T);
            if (tensorValue)
            {
                return GetConst(tensorValue.ValueUnsafe());
            }

            var floatValue = GetAttr(op, "value_float", AttributeType.Float, x => x.F);
            if (floatValue)
            {
                return Const.FromScalar(floatValue.Value());
            }

            var floatsValue = GetAttr(op, "value_floats", AttributeType.Floats, x => x.Floats);
            if (floatsValue)
            {
                var floats = floatsValue.ValueUnsafe();
                return Const.FromSpan<float>(floats.ToArray(), new Shape(floats.Count));
            }

            var intValue = GetAttr(op, "value_int", AttributeType.Int, x => x.I);

            if (intValue)
            {
                return Const.FromScalar(intValue.Value());
            }

            var intsValue = GetAttr(op, "value_ints", AttributeType.Ints, x => x.Ints);

            if (intsValue)
            {
                var ints = intsValue.ValueUnsafe();
                return Const.FromSpan<long>(ints.ToArray(), new Shape(ints.Count));
            }

            throw new NotSupportedException("Constant field format is not supported.");
        }
    }
}