// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitL2Normalization(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            return F.NN.L2Normalization(input);
        }
    }
}
