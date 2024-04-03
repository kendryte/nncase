// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.CPU;

namespace Nncase.Evaluator.IR.CPU;

public sealed class Im2colEvaluator : ITypeInferencer<Im2col>, ICostEvaluator<Im2col>, IEvaluator<Im2col>
{
    public IRType Visit(ITypeInferenceContext context, Im2col target)
    {
        var inputType = context.GetArgumentType(target, Im2col.Input);
        return inputType switch
        {
            DistributedType dt => Visit(dt, target),
            TensorType tt => Visit(tt, target),
            _ => inputType,
        };
    }

    private IRType Visit(DistributedType dt, Im2col target)
    {
        if (Visit(dt.TensorType, target) is not TensorType tensorType)
        {
            return new InvalidType("im2col typeinfer failed");
        }

        var outShape = tensorType.Shape.ToValueArray();
        var ndsbp = new SBP[dt.NdSBP.Count];
        for (int i = 0; i < dt.NdSBP.Count; i++)
        {
            var sbp = dt.NdSBP[i];
            switch (sbp)
            {
                case SBPSplit { Axis: int axis }:
                    if (axis == 0)
                    {
                        outShape[1] /= dt.Placement.Hierarchy[i];
                        ndsbp[i] = SBP.S(1);
                    }
                    else if (axis == 1)
                    {
                        outShape[0] /= dt.Placement.Hierarchy[i];
                        ndsbp[i] = SBP.S(0);
                    }
                    else
                    {
                        return new InvalidType($"can't split the axis {axis}");
                    }

                    break;
                case SBPPartialSum:
                    return new InvalidType($"can't be partial sum");
                default:
                    ndsbp[i] = sbp;
                    break;
            }
        }

        return new DistributedType(tensorType, ndsbp, dt.Placement);
    }

    private IRType Visit(TensorType tt, Im2col target)
    {
        int height = tt.Shape[2].FixedValue;
        int width = tt.Shape[3].FixedValue;
        int pad_h_before = target.Padding[0];
        int pad_h_after = target.Padding[1];
        int pad_w_before = target.Padding[2];
        int pad_w_after = target.Padding[3];
        int kernel_h = target.Kernel[0];
        int kernel_w = target.Kernel[1];
        int stride_h = target.Stride[0];
        int stride_w = target.Stride[1];
        int output_h = ((height + pad_h_before + pad_h_after -
                ((1 * (kernel_h - 1)) + 1)) / stride_h) + 1;
        int output_w = ((width + pad_w_before + pad_w_after -
         ((1 * (kernel_w - 1)) + 1)) / stride_w) + 1;
        return tt with { Shape = new Dimension[] { tt.Shape[1] * kernel_h * kernel_w, tt.Shape[0] * output_h * output_w } };
    }

    public Cost Visit(ICostEvaluateContext context, Im2col target) => new Cost()
    {
        [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, Im2col.Input)),
        [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(context.GetArgumentType<IRType>(target, Im2col.Input)),
    };

    public IValue Visit(IEvaluateContext context, Im2col target)
    {
        var inputTensor = context.GetArgumentValueAsTensor<float>(target, Im2col.Input);
        int batch = inputTensor.Shape[0].FixedValue;
        int inChannel = inputTensor.Shape[1].FixedValue;
        int height = inputTensor.Shape[2].FixedValue;
        int width = inputTensor.Shape[3].FixedValue;
        int pad_h_before = target.Padding[0];
        int pad_h_after = target.Padding[1];
        int pad_w_before = target.Padding[2];
        int pad_w_after = target.Padding[3];
        int kernel_h = target.Kernel[0];
        int kernel_w = target.Kernel[1];
        int stride_h = target.Stride[0];
        int stride_w = target.Stride[1];
        int output_h = ((height + pad_h_before + pad_h_after -
                ((1 * (kernel_h - 1)) + 1)) / stride_h) + 1;
        int output_w = ((width + pad_w_before + pad_w_after -
         ((1 * (kernel_w - 1)) + 1)) / stride_w) + 1;
        var outputTensor = Tensor.FromScalar<float>(0, new[] { inChannel * kernel_h * kernel_w, batch * output_h * output_w });

        var inputSpan = inputTensor.Buffer.Span;
        var outputSpan = outputTensor.Buffer.Span;
        var data_col = 0;
        for (int ic = 0; ic < inChannel; ic++)
        {
            for (int kh = 0; kh < kernel_h; kh++)
            {
                for (int kw = 0; kw < kernel_w; kw++)
                {
                    for (int b = 0; b < batch; b++)
                    {
                        var data_im = inputSpan.Slice((b * inChannel * height * width) + (ic * height * width));
                        int ih = -pad_h_before + kh;
                        for (int oh = 0; oh < output_h; oh++)
                        {
                            int iw = -pad_w_before + kw;
                            for (int ow = 0; ow < output_w; ow++)
                            {
                                if (iw >= 0 && iw < width && ih >= 0 && ih < height)
                                {
                                    outputSpan[data_col++] = data_im[(ih * width) + iw];
                                }
                                else
                                {
                                    outputSpan[data_col++] = 0;
                                }

                                iw += stride_w;
                            }

                            ih += stride_h;
                        }
                    }
                }
            }
        }

        return Value.FromTensor(outputTensor);
    }
}
