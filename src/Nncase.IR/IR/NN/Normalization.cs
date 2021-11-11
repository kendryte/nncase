using System;

namespace Nncase.IR.NN
{
    public sealed record L2Normalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input)
        {
          return input;
        }
    }
}