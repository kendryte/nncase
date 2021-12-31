// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
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
        private Expr VisitConv2DTranspose(in NodeProto op)
        {
            var (input, weights) = GetInputExprs(op, 0, 1);
            var bias = GetBias(op, weights, true);
            var strides = GetStrideAttribute(op);
            var dilation = GetDilationsAttribute(op);
            var group = GetIntAttribute(op, "group", 1);
            var autoPad = GetStringAttribute(op, "auto_pad", "NOTSET");
            //If "output_shape" is explicitly provided, "output_padding" does not contribute additional size to "output_shape"
            var outputPadding = GetIntsAttribute(op, "output_paddings", new[] { 0, 0 });
            // If output_shape is specified, pads values are ignored.
            
            var outputShape = GetOptionIntsAttribute(op, "output_shape");
            // 1.ignored pads
            // 2.output_padding not effect
            if (outputShape)
            {
                // compute outShape
            }
            else
            {
                
            }

            var outShape = 1;
            var pads = AutoPad(op, autoPad, input, weights, strides.ToArray<long>(), dilation.ToArray<long>());
            return F.NN.Conv2DTranspose(input, weights, bias, outShape, strides, pads, dilation, PadMode.Constant, group);
        }
    }
}
