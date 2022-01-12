// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    /// <summary>
    ///  pad tensor, a little difference with pytroch pad.
    /// </summary>
    /// <param name="PadMode"></param>
    public sealed record Pad(PadMode PadMode) : Op
    {
        /// <summary>
        /// input
        /// </summary>
        public static ParameterInfo Input = new(typeof(Pad), 0, "input");

        /// <summary>
        /// pads , shape is [channels, 2], eg. [[1,1], 
        ///                                     [2,2]]  mean pad shape [1,2,3,4] =>  [1,2,5,8].
        /// </summary>
        public static ParameterInfo Pads = new(typeof(Pad), 1, "pads", HasRank(2) & IsIntegral());

        /// <summary>
        /// float pad value
        /// </summary>
        public static ParameterInfo Value = new(typeof(Pad), 2, "value", IsScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType pads, TensorType value)
        {
            var paddings = context.GetArgument(this, Pads);
            return TypeInference.PadType(input, paddings);
        }
    }
}
