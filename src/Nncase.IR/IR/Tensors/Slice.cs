// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Slice expression.
    /// </summary>
    public sealed record Slice() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Slice), 0, "input");

        /// <summary>
        /// Gets begins.
        /// </summary>
        public static readonly ParameterInfo Begins = new(typeof(Slice), 1, "begins", IsIntegral() & HasRank(1));

        /// <summary>
        /// Gets ends.
        /// </summary>
        public static readonly ParameterInfo Ends = new(typeof(Slice), 2, "ends", IsIntegral() & HasRank(1));

        /// <summary>
        /// Gets axes.
        /// </summary>
        public static readonly ParameterInfo Axes = new(typeof(Slice), 3, "axes", IsIntegral() & HasRank(1));

        /// <summary>
        /// Gets strides.
        /// </summary>
        public static readonly ParameterInfo Strides = new(typeof(Slice), 4, "strides", IsIntegral() & HasRank(1));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context,
          TensorType input, TensorType begins, TensorType ends,
           TensorType axes, TensorType strides)
        {
            if (context.GetArgument(this, Begins) is Const begins_con &&
                context.GetArgument(this, Ends) is Const ends_con &&
                context.GetArgument(this, Axes) is Const axes_con &&
                context.GetArgument(this, Strides) is Const strides_con
                )
            {
                var outshape = new List<Dimension>();
                var ts_begins = begins_con.ToTensor<int>();
                var ts_ends = ends_con.ToTensor<int>();
                var ts_strides = strides_con.ToTensor<int>();
                foreach (var axis in axes_con.ToTensor<int>())
                {
                    var begin = ts_begins[axis];
                    var end = ts_ends[axis];
                    var stride = ts_strides[axis];
                    if (input.Shape[axis].IsFixed)
                    {
                        var old = input.Shape[axis].FixedValue;
                        begin = begin >= 0 ? begin : old + begin;
                        end = end >= 0 ? end : old + begin;
                        stride = stride >= 0 ? stride : -stride;
                        outshape.Add((end - begin) / stride);
                    }
                    else
                        outshape.Add(Dimension.Unknown);
                }
                return input with { Shape = new Shape(outshape) };
            }
            return new InvalidType("Can't Infer Shape With Dynamic Input!");
        }
    }
}
