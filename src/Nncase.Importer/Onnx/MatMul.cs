﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitMatMul(in NodeProto op)
        {
            var (a, b) = GetInputExprs(op, 0, 1);
            var matmul = IR.F.Math.MatMul(a, b);
            List<string> outputNames = new() { op.Output[0] };
            matmul.Metadata.OutputNames = outputNames;
            return matmul;
        }
    }
}
