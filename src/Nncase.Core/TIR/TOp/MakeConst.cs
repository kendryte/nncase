using Nncase.IR;
using System;
namespace Nncase.TIR
{

    public sealed record MakeConst<T>(T Value) : Op
    {
        public static readonly ParameterInfo Lanes = new(typeof(MakeConst<T>), 0, "lanes");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType lanes)
        {
            return new InvalidType("This Node Should Be Fold!");
        }
    }
}