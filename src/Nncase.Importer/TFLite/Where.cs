// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitWhere(in tflite.Operator op)
        {
            var cond = GetInputExprs(op, 0);
            var x = op.InputsLength >= 2 ? GetInputExprs(op, 1) : cond;
            var y = op.InputsLength >= 3 ? GetInputExprs(op, 2) : cond;
            return Where(cond, x, y);
        }
    }
}