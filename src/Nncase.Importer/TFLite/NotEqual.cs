// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Diagnostics;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using tflite;
using static Nncase.IR.F.Math;
using F = Nncase.IR.F;
using TensorType = tflite.TensorType;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitNotEqual(in tflite.Operator op)
        {
            var (lhs, rhs) = GetInputExprs(op, 0, 1);
            return Compare(CompareOp.NotEqual, lhs, rhs);
        }
    }
}
