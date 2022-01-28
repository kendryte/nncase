// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace Nncase.IR.Tensors
{
    /// <summary>
    /// Flatten expression.
    /// </summary>
    public sealed record Flatten() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Flatten), 0, "input");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(Flatten), 1, "axis");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis)
        {
            if (context.GetArgument(this, Axis) is Const axisV)
            {
                if (input.Shape.IsFixed)
                {
                    var axisValue = axisV.ToScalar<int>();
                    var first = input.Shape.Take(axisValue).Aggregate(1, (x, y) => x * y.FixedValue);
                    var second = input.Shape.Take(axisValue..input.Shape.Count).Aggregate(1, (x, y) => x * y.FixedValue);
                    return input with { Shape = new[] { first, second } };
                }
                else
                {
                    return new InvalidType("Can't infer shape with dynamic input in Flatten");
                }
            }
            else
            {
                return new InvalidType("Can't infer shape with dynamic axis in Flatten");
            }
        }
    }
}