﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
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
            var perm = Tensor.From<long>(GetIntsAttribute(op, "perm"));
            return F.Tensors.Transpose(input, perm);
        }
    }
}
