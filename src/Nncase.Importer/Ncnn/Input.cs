﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Options;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.F;
using Nncase.IR.NN;

namespace Nncase.Importer.Ncnn;

public partial class NcnnImporter
{
    private Expr VisitInput(NcnnLayer layer)
    {
        var w = layer.ParamDict.GetInt(0) ?? (Expr)Var.SizeVar("w");
        var h = layer.ParamDict.GetInt(1) ?? (Expr)Var.SizeVar("h");
        var c = layer.ParamDict.GetInt(2) ?? (Expr)Var.SizeVar("c");
        var shape = new Shape(c, h, w);
        var input = new Var(layer.Name, new TensorType(DataTypes.Float32, shape));
        _inputs.Add(input);
        return input;
    }
}
