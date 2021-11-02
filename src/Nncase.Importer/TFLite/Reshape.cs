// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitReshape(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var output = GetOutputTensor(op, 0);
            var outShape = new Const(
                GetIRType(new[] { output.ShapeLength }, output.Type),
                DataTypes.GetBytes<int>(output.GetShapeBytes()));
            return F.Tensors.Reshape(input, outShape);
        }
    }
}