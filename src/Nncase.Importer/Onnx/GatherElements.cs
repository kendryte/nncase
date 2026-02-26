// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitGatherElements(in NodeProto op)
        {
            var (input, indices) = GetInputExprs(op, 0, 1);
            var axis = GetIntAttribute(op, "axis", 0);
            if (axis < 0)
            {
                axis += input.CheckedShape.Rank;
            }

            return F.Tensors.GatherElements(input, axis, indices);
        }
    }
}
