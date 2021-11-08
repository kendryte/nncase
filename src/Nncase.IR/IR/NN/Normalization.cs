using System;

namespace Nncase.IR.NN
{
    public sealed record L2Normalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Sigmoid), 0, "input");

        /// <inheritdoc/>
        public override IRType InferInvokeResultType(ITypeInferenceContext context)
        {
            throw new NotImplementedException();
        }
    }
}