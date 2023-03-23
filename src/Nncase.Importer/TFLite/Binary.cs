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
        private object VisitBinary(in tflite.Operator op, BinaryOp binaryOp, tflite.ActivationFunctionType activation = tflite.ActivationFunctionType.NONE)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);

            lhs.GetMetadata().SetOutPutsNames(new List<string> { GetInputTensor(op, 0).Name });
            rhs.GetMetadata().SetOutPutsNames(new List<string> { GetInputTensor(op, 1).Name });

            var node = F.Math.Binary(binaryOp, lhs, rhs);
            return Activate(node, activation);
        }

        private Expr VisitFloorDiv(in tflite.Operator op)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);

            lhs.GetMetadata().SetOutPutsNames(new List<string> { GetInputTensor(op, 0).Name });
            rhs.GetMetadata().SetOutPutsNames(new List<string> { GetInputTensor(op, 1).Name });

            return F.Math.FloorDiv(lhs, rhs);
        }

        private Expr VisitFloorMod(in tflite.Operator op)
        {
            (var lhs, var rhs) = GetInputExprs(op, 0, 1);

            lhs.GetMetadata().SetOutPutsNames(new List<string> { GetInputTensor(op, 0).Name });
            rhs.GetMetadata().SetOutPutsNames(new List<string> { GetInputTensor(op, 1).Name });

            return F.Math.FloorMod(lhs, rhs);
        }
    }
}
