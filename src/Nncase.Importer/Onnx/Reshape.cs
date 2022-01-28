// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitReshape(in NodeProto op)
        {
            var (input, shape) = GetInputExprs(op, 0, 1);

            // allowzero has been avaliable since opset 14
            var allowZero = GetBoolAttribute(op, "allowzero", false);
            if (allowZero)
            {
                throw new NotSupportedException("Not support reshape attribute: allowzero");
            }

            return F.Tensors.Reshape(input, shape);
        }
    }
}