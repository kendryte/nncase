// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitShape(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);

            // DType of ShapeOf in TF is int32
            // but in onnx it's int64
            // TF op expect a int32 result and compute with other i32 data
            return Cast(ShapeOf(input), DataTypes.Int32);
        }
    }
}
