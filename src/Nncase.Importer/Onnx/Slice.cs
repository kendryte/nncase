// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitSlice(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            var (starts, ends) = GetInputExprs(op, 1, 2);
            var axes = GetOptionInputExpr(op, 3).Or(0);
            var steps = GetOptionInputExpr(op, 4).Or(1);
            return F.Tensors.Slice(input, starts, ends, axes, steps);
        }
    }
}