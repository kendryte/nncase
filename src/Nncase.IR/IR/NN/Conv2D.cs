// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.NN
{
    /// <summary>
    /// Conv2D.
    /// </summary>
    public sealed record Conv2D(PadMode PadMode) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Conv2D), 0, "input", HasRank(4));

        /// <summary>
        /// Gets Weights.
        /// </summary>
        public static readonly ParameterInfo Weights = new(typeof(Conv2D), 1, "weights", HasRank(4));

        /// <summary>
        /// Gets Bias.
        /// </summary>
        public static readonly ParameterInfo Bias = new(typeof(Conv2D), 2, "bias", HasRank(1));

        /// <summary>
        /// Gets Stride.
        /// </summary>
        public static readonly ParameterInfo Stride = new(typeof(Conv2D), 3, "stride", HasRank(1) & IsIntegral());

        /// <summary>
        /// Gets Padding.
        /// </summary>
        public static readonly ParameterInfo Padding = new(typeof(Conv2D), 4, "padding", HasRank(2) & IsIntegral());

        /// <summary>
        /// Gets Dilation.
        /// </summary>
        public static readonly ParameterInfo Dilation = new(typeof(Conv2D), 5, "dilation", HasRank(1) & IsIntegral());

        /// <summary>
        /// Gets Groups.
        /// </summary>
        public static readonly ParameterInfo Groups = new(typeof(Conv2D), 6, "groups", IsScalar() & IsIntegral());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context,
          TensorType input, TensorType weights, TensorType bias,
          TensorType stride, TensorType padding, TensorType dilation, TensorType groups)
        {
            var outshape = input.Shape.ToList();
            outshape[1] = weights.Shape[0];
            if (
            context.GetArgument(this, Stride) is Const stride_con &&
            context.GetArgument(this, Padding) is Const padding_con &&
            context.GetArgument(this, Dilation) is Const dilation_con &&
            context.GetArgument(this, Groups) is Const groups_con &&
            input.Shape[2].IsFixed &&
            input.Shape[3].IsFixed &&
            weights.Shape[2].IsFixed &&
            weights.Shape[3].IsFixed
            )
            {
                var ts_stride = stride_con.ToTensor<int>();
                var ts_padding = padding_con.ToTensor<int>();
                var ts_dilation = dilation_con.ToTensor<int>();
                var groups_v = groups_con.ToScalar<int>();

                outshape[2] = GetWindowedOutputSize(input.Shape[2].FixedValue + ts_stride[0, 0] + ts_stride[0, 1],
                  weights.Shape[2].FixedValue, ts_stride[0], ts_dilation[0], false);
                outshape[3] = GetWindowedOutputSize(input.Shape[3].FixedValue + ts_stride[1, 0] + ts_stride[1, 1],
                  weights.Shape[3].FixedValue, ts_stride[1], ts_dilation[1], false);
            }
            else
            {
                outshape[2] = outshape[3] = Dimension.Unknown;
            }
            return input with { Shape = new Shape(outshape) };
        }
    }
}
