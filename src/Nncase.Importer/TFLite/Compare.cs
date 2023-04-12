// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitCompare(in tflite.Operator op, CompareOp compareOp)
        {
            var (x, y) = GetInputExprs(op, 0, 1);
            return F.Math.Compare(compareOp, x, y);
        }
    }
}
