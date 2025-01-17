// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.Math;
using TorchSharp;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class HuggingFaceImporter
    {
        private Expr VisitQwen2Model(Dictionary<string, object> modelConfig, Dictionary<string, Tensor> constTensors)
        {
            // var (lhs, rhs) = GetInputExprs(op, 0, 1);
            // if (binaryOp == BinaryOp.Pow && lhs.CheckedDataType != rhs.CheckedDataType)
            // {
            //     return F.Math.Binary(binaryOp, lhs, IR.F.Tensors.Cast(rhs, lhs.CheckedDataType)).With(metadata: new IRMetadata() { OutputNames = op.Output });
            // }

            // return F.Math.Binary(binaryOp, lhs, rhs).With(metadata: new IRMetadata() { OutputNames = op.Output });

            var input_ids = new Var();
            var position_ids = new Var();

        }
    }
}
