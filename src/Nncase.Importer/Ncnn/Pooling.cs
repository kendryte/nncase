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

namespace Nncase.Importer.Ncnn;

public partial class NcnnImporter
{
    private Expr VisitPooling(NcnnLayer layer)
    {
        var poolingType = layer.ParamDict.Get(0, 0);
        var kernelW = layer.ParamDict.Get(1, 0);
        var kernelH = layer.ParamDict.Get(11, kernelW);
        var strideW = layer.ParamDict.Get(2, 1);
        var strideH = layer.ParamDict.Get(12, strideW);
        var padLeft = layer.ParamDict.Get(3, 0);
        var padRight = layer.ParamDict.Get(14, padLeft);
        var padTop = layer.ParamDict.Get(13, padLeft);
        var padBottom = layer.ParamDict.Get(15, padTop);
        var globalPooling = layer.ParamDict.Get(4, 0) != 0;
        var padMode = layer.ParamDict.Get(5, 0);
        var avgpoolCountIncludePad = layer.ParamDict.Get(6, 0) != 0;
        var adaptivePooling = layer.ParamDict.Get(7, 0) != 0;
        var outW = layer.ParamDict.Get(8, 0);
        var outH = layer.ParamDict.Get(18, outW);

        Expr pooling;
        var input = CHWToNCHW(GetInputExprs(layer, 0));
        (var reduceOp, var initValue) = poolingType switch
        {
            0 => (ReduceOp.Max, float.NegativeInfinity),
            1 => (ReduceOp.Mean, 0f),
            _ => throw new NotSupportedException($"Unsupported pooling type: {poolingType}."),
        };
        var filter = Tensor.From(new[] { kernelH, kernelW }, new[] { 2 });
        var stride = Tensor.From(new[] { strideH, strideW }, new[] { 2 });
        var dilation = Tensor.FromScalar(0, new[] { 2, 2 });

        if (globalPooling)
        {
            pooling = Tensors.Reduce(reduceOp, input, new[] { 2, 3 }, initValue, false);
        }
        else if (adaptivePooling)
        {
            var padding = Tensor.FromScalar(0, new[] { 2, 2 });
            var inShape = Tensors.ShapeOf(input);
            var w = inShape[3];
            var h = inShape[2];
            var kernelExtentH = h - outH + 1;
            var kernelExtentW = w - outW + 1;
            var adaptiveFilter = Tensors.Stack(new IR.Tuple(kernelExtentH, kernelExtentW), 0);
            var adaptiveStride = Tensor.FromScalar(1, 2);

            pooling = NN.ReduceWindow2D(reduceOp, input, initValue, adaptiveFilter, adaptiveStride, padding, dilation, false, avgpoolCountIncludePad);
        }
        else
        {
            Expr[] paddingH;
            Expr[] paddingW;

            if (padMode == 1)
            {
                // valid padding
                paddingH = new Expr[] { padTop, padBottom };
                paddingW = new Expr[] { padLeft, padRight };
            }
            else
            {
                var inShape = Tensors.ShapeOf(input);
                var w = inShape[3];
                var h = inShape[2];

                if (padMode == 0)
                {
                    // full padding
                    var tailW = (w + padLeft + padRight - kernelW) % strideW;
                    var tailH = (h + padTop + padBottom - kernelH) % strideH;

                    var tailPadW = IR.F.Math.Select(IR.F.Math.Equal(tailW, 0), 0, tailW);
                    var tailPadH = IR.F.Math.Select(IR.F.Math.Equal(tailH, 0), 0, tailH);

                    paddingH = new Expr[] { padTop, padBottom + tailPadH };
                    paddingW = new Expr[] { padLeft, padRight + tailPadW };
                }
                else if (padMode is 2 or 3)
                {
                    // valid padding
                    var padH = kernelH + ((h - 1) / strideH * strideH) - h;
                    var padW = kernelW + ((w - 1) / strideW * strideW) - w;

                    if (padMode == 2)
                    {
                        // tensorflow padding=SAME or onnx padding=SAME_UPPER
                        paddingH = new Expr[] { padH / 2, padH - (padH / 2) };
                        paddingW = new Expr[] { padW / 2, padW - (padW / 2) };
                    }
                    else
                    {
                        // onnx padding=SAME_LOWER
                        paddingH = new Expr[] { padH - (padH / 2), padH / 2 };
                        paddingW = new Expr[] { padW - (padW / 2), padW / 2 };
                    }
                }
                else
                {
                    throw new NotSupportedException($"Unsupported pad mode: {padMode}.");
                }
            }

            var padding = Util.ConcatPadding(paddingH, paddingW);
            pooling = NN.ReduceWindow2D(reduceOp, input, initValue, filter, stride, padding, dilation, false, avgpoolCountIncludePad);
        }

        var output = NCHWToCHW(pooling);
        return output;
    }
}
