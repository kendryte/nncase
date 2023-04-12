// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSquareDifference(in tflite.Operator op)
        {
            var (input1, input2) = GetInputExprs(op, 0, 1);
            var val = input1 - input2;
            return val * val;
        }
    }
}
