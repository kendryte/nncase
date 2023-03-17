// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr SetOutputsNames(Expr expr, in NodeProto op)
        {
            for (int i = 0; i < op.Output.Count; i++)
            {
                expr.OutputsNames.Add(op.Output[i]);
            }

            return expr;
        }
    }
}
