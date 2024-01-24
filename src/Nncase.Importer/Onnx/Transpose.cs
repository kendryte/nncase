// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using DryIoc.ImTools;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitTranspose(NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            var defaultPerm = Enumerable.Range(0, input.CheckedShape.Rank).Reverse().ToArray();
            var perm = Tensor.From(GetIntsAttribute(op, "perm", defaultPerm));
            return F.Tensors.Transpose(input, perm).With(metadata: new IRMetadata() { OutputNames = op.Output, });
        }
    }
}
