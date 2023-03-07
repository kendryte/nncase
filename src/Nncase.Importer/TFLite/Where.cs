// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using Nncase.IR;
using static Nncase.IR.F.Tensors;
using Where = Nncase.IR.Tensors.Where;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitWhere(in tflite.Operator op)
        {
            var cond = GetInputExprs(op, 0);
            return op.InputsLength switch
            {
                1 => Where(cond, Array.Empty<float>(), Array.Empty<float>(), true),
                3 => Where(cond, GetInputExprs(op, 1), GetInputExprs(op, 2), true),
                _ => throw new NotImplementedException("Not Impl for where which has 2 input"),
            };
        }
    }
}
