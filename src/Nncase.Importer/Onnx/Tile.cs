// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitTile(in NodeProto op)
        {
            var (input, repeats) = GetInputExprs(op, 0, 1);
            return Tile(input, repeats);
        }
    }
}
