// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;

namespace Nncase.IR
{
    /// <summary>
    /// Type inference helper.
    /// </summary>
    public static class TypeInference
    {
        /// <summary>
        /// Check argument type.
        /// </summary>
        /// <typeparam name="T">Desired type.</typeparam>
        /// <param name="context">Type inference context.</param>
        /// <param name="op">Operator.</param>
        /// <param name="parameter">Parameter.</param>
        /// <param name="reason">Reason text if not satisfied.</param>
        /// <returns>The desired type.</returns>
        public static T CheckArgumentType<T>(this ITypeInferenceContext context, Op op, ParameterInfo parameter, string? reason = null)
            where T : IRType
        {
            return context.GetArgumentType(op, parameter) switch
            {
                T t => t,
                AnyType a => throw new TypeInferenceInterruptException(a),
                _ => throw new TypeInferenceInterruptException(new InvalidType(reason ?? $"{op}.{parameter.Name} must be a {typeof(T).Name}.")),
            };
        }

        /// <summary>
        /// Throw <seealso cref="TypeInferenceInterruptException"/> if type is <seealso cref="AnyType"/> or <seealso cref="InvalidType"/>.
        /// </summary>
        /// <typeparam name="T">Type.</typeparam>
        /// <param name="t">Type instance.</param>
        /// <returns>Original type instance.</returns>
        public static T ThrowIfTypeInferenceInterrupt<T>(this T t)
            where T : IRType
        {
            return t switch
            {
                AnyType a => throw new TypeInferenceInterruptException(a),
                InvalidType i => throw new TypeInferenceInterruptException(i),
                T other => other,
            };
        }

        /// <summary>
        /// Broadcast input shapes.
        /// </summary>
        /// <param name="inputs">Input shapes.</param>
        /// <returns>Broadcasted shape.</returns>
        public static IRType BroadcastType(params TensorType[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new ArgumentException("Broadcast must have 2 inputs at least.");
            }

            var dataType = inputs[0].DataType;
            if (inputs.Any(x => x.DataType != dataType))
            {
                return new InvalidType("Inputs of broadcast must have same datatype.");
            }

            // If any input is invalid, result is invalid
            if (inputs.Any(x => x.Shape.IsInvalid))
            {
                return TensorType.Invalid(dataType);
            }

            // If any input is not fixed, result is unranked
            if (inputs.Any(x => !x.Shape.IsFixed))
            {
                return TensorType.Unranked(dataType);
            }

            var outputRank = inputs.Select(x => x.Shape.Rank).Max();
            var outputShape = new Dimension[outputRank];
            Span<long> inputDims = stackalloc long[inputs.Length];

            for (int dimIndex = 0; dimIndex < outputShape.Length; dimIndex++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var inShape = inputs[i].Shape;
                    var inExtend = outputRank - inShape.Rank;
                    var inDimIndex = dimIndex - inExtend;
                    var inDim = inDimIndex < 0 ? 1 : inShape[inDimIndex].Value!.Value;
                    Debug.Assert(inDim != 0, "Input dimension should not be 0.");
                    inputDims[i] = inDim;
                }

                // 1. Sort descending
                inputDims.Sort((a, b) => b.CompareTo(a));

                // 2. Find first 1
                var firstOneIndex = inputDims.IndexOf(1);
                var expectedDim = inputDims[0];

                // 3. Dims before 1 are all same or 1 is not found, it's ok to broadcast
                if (firstOneIndex == -1 ||
                    inputDims[..firstOneIndex].AsValueEnumerable().All(x => x == expectedDim))
                {
                    outputShape[dimIndex] = expectedDim;
                }
                else
                {
                    return new InvalidType("Inputs are not compatible to broadcast.");
                }
            }

            return new TensorType(dataType, new Shape(outputShape));
        }
    }
}
