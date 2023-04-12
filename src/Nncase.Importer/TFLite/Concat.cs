// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitConcat(in tflite.Operator op)
        {
            var @operator = op;
            var inputs = Enumerable.Range(0, op.InputsLength).Select(i => GetInputExprs(@operator, i)).ToArray();
            return F.Tensors.Concat(new Tuple(inputs), op.BuiltinOptionsAsConcatenationOptions().Axis);
        }

        private Expr VisitPack(in tflite.Operator op)
        {
            var @operator = op;
            var axis = op.BuiltinOptionsAsPackOptions().Axis;
            var inputs = Enumerable.Range(0, op.InputsLength).Select(i => GetInputExprs(@operator, i)).ToArray();
            return F.Tensors.Stack(new Tuple(inputs), axis);
        }
    }
}
