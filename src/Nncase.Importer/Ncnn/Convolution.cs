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

namespace Nncase.Importer.Ncnn;

public partial class NcnnImporter
{
    private Expr VisitConvolution(NcnnLayer layer)
    {
        var numOutput = layer.ParamDict.Get(0, 0);
        var kernelW = layer.ParamDict.Get(1, 0);
        var kernelH = layer.ParamDict.Get(11, kernelW);
        var dilationW = layer.ParamDict.Get(2, 1);
        var dilationH = layer.ParamDict.Get(12, dilationW);
        var strideW = layer.ParamDict.Get(3, 1);
        var strideH = layer.ParamDict.Get(13, strideW);
        var padLeft = layer.ParamDict.Get(4, 0);
        var padRight = layer.ParamDict.Get(15, padLeft);
        var padTop = layer.ParamDict.Get(14, padLeft);
        var padBottom = layer.ParamDict.Get(16, padTop);
        var padValue = layer.ParamDict.Get(18, 0.0f);
        var biasTerm = layer.ParamDict.Get(5, 0);
        var weightDataSize = layer.ParamDict.Get(6, 0);
        var group = layer.ParamDict.Get(7, 1);
        var activationType = layer.ParamDict.Get(9, 0);
        var activationParams = layer.ParamDict.Get(10, Tensor<float>.Empty).Buffer.Span;

        var numInput = weightDataSize / numOutput / kernelW / kernelH;
        var kernelExtentW = (dilationW * (kernelW - 1)) + 1;
        var kernelExtentH = (dilationH * (kernelH - 1)) + 1;

        if (padValue != 0)
        {
            throw new NotSupportedException($"Unsupported pad value: {padValue}.");
        }

        var input = CHWToNCHW(GetInputExprs(layer, 0));

        Expr[] paddingH;
        Expr[] paddingW;

        if (padLeft is -233 or -234)
        {
            var inShape = Tensors.ShapeOf(input);
            var w = inShape[3];
            var h = inShape[2];
            var padW = kernelExtentW + ((w - 1) / strideW * strideW) - w;
            var padH = kernelExtentH + ((h - 1) / strideH * strideH) - h;

            if (padLeft == -233)
            {
                // SAME_UPPER
                paddingH = new[] { padH / 2, padH - (padH / 2) };
                paddingW = new[] { padW / 2, padW - (padW / 2) };
            }
            else
            {
                // SAME_LOWER
                paddingH = new[] { padH - (padH / 2), padH / 2 };
                paddingW = new[] { padW - (padW / 2), padW / 2 };
            }
        }
        else
        {
            paddingH = new Expr[] { padTop, padBottom };
            paddingW = new Expr[] { padLeft, padRight };
        }

        var stride = Tensor.From(new[] { strideH, strideW }, new[] { 2 });
        var dilation = Tensor.From(new[] { dilationH, dilationW }, new[] { 2 });
        var clampRange = ToFloatClampRange(activationType, activationParams);
        var clamp = Tensor.From(new[] { clampRange.Min, clampRange.Max }, new[] { 2 });
        var padding = Util.ConcatPadding(paddingH, paddingW);
        var weights = _modelBin.LoadFloat32(new[] { numOutput, numInput, kernelH, kernelW }, true);
        var bias = biasTerm != 0 ? _modelBin.LoadFloat32(new[] { numOutput }, false) : Tensor.FromScalar(0f, numOutput);

        var conv2d = NN.Conv2D(input, weights, bias, stride, padding, dilation, PadMode.Constant, group, clamp);
        conv2d = activationType switch
        {
            0 or 1 or 3 => conv2d,
            2 => NN.LeakyRelu(conv2d, activationParams[0]),
            4 => NN.Sigmoid(conv2d),
            5 => NN.Mish(conv2d),
            _ => throw new NotSupportedException($"Unsupported activation type: {activationType}."),
        };
        var output = NCHWToCHW(conv2d);
        return output;
    }
}
