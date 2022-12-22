// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSqueeze(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var options = op.BuiltinOptionsAsSqueezeOptions();
            var dims = options.GetSqueezeDimsArray();
            return Squeeze(input, dims);
        }

        private Expr VisitExpandDims(in tflite.Operator op)
        {
            var (input, dim) = GetInputExprs(op, 0, 1);
            return Unsqueeze(input, Unsqueeze(dim, new[] { 0 }));
        }
    }
}
