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
        private Expr VisitSoftmax(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            var axis = GetIntAttribute(op, "axis", -1);
            return F.NN.SoftMax(input, axis);
        }
        
        private Expr VisitLogSoftmax(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axis = GetIntAttribute(op, "axis", -1);
            return F.NN.LogSoftMax(input, axis);
        }

        private Expr VisitSoftplus(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return F.NN.SoftPlus(input);
        }

        private Expr VisitSoftsign(in NodeProto op)
        {
            var input = GetSingleInputExpr(op);
            return F.NN.SoftSign(input);
        }
    }
}