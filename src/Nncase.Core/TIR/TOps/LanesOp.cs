using Nncase.IR;
using System;
namespace Nncase.TIR
{
    public sealed record LanesOp() : Op
    {
        public static readonly ParameterInfo Input = new(typeof(LanesOp), 0, "input");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input)
        {
            return new TensorType(DataType.Int32, Shape.Scalar);
        }
    }
}