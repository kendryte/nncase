// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using static Nncase.Util;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSplit(in tflite.Operator op)
        {
            var axis = GetInputExprs<Dimension>(op, 0);
            var input = GetInputExprs<Expr>(op, 1);
            var splits = op.BuiltinOptionsAsSplitOptions().NumSplits;
            var a = axis.FixedValue;
            var s = ComputeSplit(input, splits, a);
            _ = GetOutputTensor(op, 0);
            return F.Tensors.Split(input, axis, s);
        }
    }
}
