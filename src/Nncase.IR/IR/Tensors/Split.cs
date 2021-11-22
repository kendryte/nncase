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
        public static readonly ParameterInfo Axis = new(typeof(Split), 1, "axis", IsScalar() & IsIntegral());

        /// <summary>
        /// Gets sections.
        /// </summary>
        public static readonly ParameterInfo Sections = new(typeof(Split), 2, "sections", IsIntegral() & HasRank(1));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis, TensorType sections)
        {
            if (context.GetArgument(this, Axis) is Const axis_con &&
                context.GetArgument(this, Sections) is Const sections_con)
            {
                var axis_v = axis_con.ToScalar<int>();
                var sections_v = sections_con.ToTensor<int>();
                var inshape = input.Shape.ToArray();
                if (inshape[axis_v] == Dimension.Unknown)
                    return new InvalidType("The Input Shape Axis Can Not Be Unknown!");

                if (sections_v.Length == 1) /* split */
                {
                    if (0 != inshape[axis_v].FixedValue % sections_v[0])
                        return new InvalidType("The Section Value Not Match Shape[Axis]!");
                    var outshape = new Dimension[inshape.Length];
                    Array.Copy(inshape, outshape, inshape.Length);
                    outshape[axis_v] = new Dimension(inshape[axis_v].FixedValue / sections_v[0]);
                    return new TupleType(Enumerable.Repeat((IRType)(input with { Shape = new Shape(outshape) }), sections_v[0]));
                }
                else
                {
                    if (sections_v.Sum() != inshape[axis_v].FixedValue)
                        return new InvalidType("The Sections Sum Must Equal To Shape[Axis]!");
                    var outshape = new Dimension[inshape.Length];
                    Array.Copy(inshape, outshape, inshape.Length);
                    return new TupleType(from section in sections_v
                                         let x = (outshape[axis_v] = section)
                                         select input with { Shape = new Shape(outshape) });

                }
            }
            return new InvalidType("The Sections And Axis Must Be Const!");
        }
    }
}
