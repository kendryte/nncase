// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitCast(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var output = GetOutputTensor(op, 0);
            return Cast(input, GetDataType(output.Type));
        }
    }
}
