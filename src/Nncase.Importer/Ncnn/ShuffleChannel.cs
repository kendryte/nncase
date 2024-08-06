// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.F;
using Nncase.IR.NN;
using Nncase.Runtime.Ncnn;

namespace Nncase.Importer.Ncnn;

public partial class NcnnImporter
{
    private Expr VisitShuffleChannel(NcnnLayer layer)
    {
        var group = layer.ParamDict.Get(0, 1);
        var reverse = layer.ParamDict.Get(1, 0) != 0;

        var input = GetInputExprs(layer, 0);
        var inShape = Tensors.ShapeOf(input);
        var channels = inShape[0];
        var h = inShape[1];
        var w = inShape[2];
        var realGroup = reverse ? channels / group : (Expr)group;
        var channelsPerGroup = channels / realGroup;

        var rshape1 = Tensors.Reshape(input, Tensors.Stack(new IR.Tuple(realGroup, channelsPerGroup, h, w), 0));
        var tp = Tensors.Transpose(rshape1, new[] { 1, 0, 2, 3 });
        var output = Tensors.Reshape(tp, Tensors.Stack(new IR.Tuple(channels, h, w), 0));
        return output;
    }
}
