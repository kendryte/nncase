// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitConcat(in tflite.Operator op)
        {
            var inputs = new Expr[op.InputsLength];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = GetInputExprs(op, i);
            }
            return F.Tensors.Concat(new Tuple(inputs), op.BuiltinOptionsAsConcatenationOptions().Axis);
        }
    }
}