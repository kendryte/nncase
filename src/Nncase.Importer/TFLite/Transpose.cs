// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitTranspose(in tflite.Operator op)
        {
            var (input, perm) = GetInputExprs(op, 0, 1);
            return F.Tensors.Transpose(input, perm);
        }
    }
}
