// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.IO;
using System.Linq;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitSqueeze(in tflite.Operator op)
        {
            var input = GetInputExprs(op, 0);
            var options = op.BuiltinOptionsAsSqueezeOptions();
            var dims = options.GetSqueezeDimsArray().ToList();
            var tmp = input;
            while (dims.Any())
            {
                var dim = dims.First();
                tmp = F.Tensors.Squeeze(tmp, dim);
                dims.ForEach(d =>
                {
                    if (d >= dim)
                    {
                        d -= 1;
                    }
                });
            }

            return tmp;
        }

        private Expr VisitExpandDims(in tflite.Operator op)
        {
            var (input, dim) = GetInputExprs(op, 0, 1);
            return F.Tensors.Unsqueeze(input, dim);
        }
    }
}