using System;
using static Nncase.IR.Utility;

namespace Nncase.IR.NN
{
    public sealed record SoftMax() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(SoftMax), 0, "input");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input)
        {
            return input;
        }
    }

    public sealed record LogSoftMax() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(LogSoftMax), 0, "input");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input)
        {
            return input;
        }
    }
}