// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitBinary(in tflite.Operator op, BinaryOp binaryOp, tflite.ActivationFunctionType activation = tflite.ActivationFunctionType.NONE)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);

            var node = F.Math.Binary(binaryOp, lhs, rhs);
            return Activate(node, activation);
        }

        private Expr VisitFloorDiv(in tflite.Operator op)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);
            return F.Math.FloorDiv(lhs, rhs);
        }

        private Expr VisitFloorMod(in tflite.Operator op)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);
            return F.Math.FloorMod(lhs, rhs);
        }
    }
}
