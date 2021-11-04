// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSlice(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var (begin, size) = GetInputExprs(op, 1, 2); 
            var endValue = GetShapeDataFromConst(begin).Zip(GetShapeDataFromConst(size), (x, y) => x + y).ToArray();
            var end = new Const(((Const)begin).ValueType, DataTypes.GetBytes<int>(endValue));
            return F.Tensors.Slice(input, begin, end);
        }
    }
}