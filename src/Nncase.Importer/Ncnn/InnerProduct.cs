// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Options;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.Runtime.Ncnn;

namespace Nncase.Importer.Ncnn;

public partial class NcnnImporter
{
    private Expr VisitInnerProduct(NcnnLayer layer)
    {
        var numOutput = layer.ParamDict.Get(0, 0);
        var biasTerm = layer.ParamDict.Get(1, 0) != 0;
        var weightDataSize = layer.ParamDict.Get(2, 0);
        var activationType = layer.ParamDict.Get(9, 0);
        var activationParams = layer.ParamDict.Get(10, Tensor<float>.Empty).Buffer.Span;

        var numInput = weightDataSize / numOutput;

        var input = Tensors.Unsqueeze(GetInputExprs(layer, 0), new[] { 0 });
        var weights = Tensors.Transpose(_modelBin.LoadFloat32(new[] { numOutput, numInput }, true), new[] { 1, 0 });

        Expr output = IR.F.Math.MatMul(input, weights);
        if (biasTerm)
        {
            output += _modelBin.LoadFloat32(new[] { numOutput }, false);
        }

        output = ApplyActivation(output, activationType, activationParams);
        output = Tensors.Squeeze(output, new[] { 0 });
        return output;
    }
}
