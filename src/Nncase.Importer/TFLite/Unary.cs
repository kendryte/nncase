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
        private Expr VisitUnary(in tflite.Operator op, UnaryOp unaryOp)
        {
            var input = GetInputExprs(op, 0);
            return F.Math.Unary(unaryOp, input);
        }
    }
}
