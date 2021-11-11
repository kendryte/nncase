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
    /// Split expression.
    /// </summary>
    public sealed record Split() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Split), 0, "input");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(Split), 1, "axis", IsScalar(IsIntegral()));

        /// <summary>
        /// Gets sections.
        /// </summary>
        public static readonly ParameterInfo Sections = new(typeof(Split), 2, "sections", IsIntegral() & HasRank(1));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis, TensorType sections)
        {
            if (!Axis.CheckType(axis))
                return new InvalidType("The Axis Must Be Scalar");
            if (!Sections.CheckType(sections))
                return new InvalidType("The Sections Rank Must Equal 1");

            if (context.GetArgument(this, Axis) is Const axis_con &&
                context.GetArgument(this, Sections) is Const sections_con)
            {
                var axis_v = axis_con.ToScalar<int>();
                var sections_v = sections_con.ToTensor<int>();

                // var dim_v = dim_con.ToScalar<int>();
                // var outshape = input.Shape.ToList();
                // if (outshape[dim_v].IsFixed && outshape[dim_v].FixedValue == 1)
                // {
                //     outshape.RemoveAt(dim_v);
                //     return input with { Shape = new Shape(outshape) };
                // }
                // return new InvalidType("The Shape[dim] is not 1!");
            }
            return new InvalidType("The Sections And Axis Must Be Const!");

        }
    }
}
