// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using Nncase.IR;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitTile(in tflite.Operator op)
        {
            var (input, multiples) = GetInputExprs(op, 0, 1);
            return Tile(input, multiples);
        }
    }
}
