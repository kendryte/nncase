using Nncase.IR;
using System;
namespace Nncase.TIR
{
    /// <summary>
    /// <see cref="F.TOps.Ramp(Expr, Expr, int)"/>
    /// </summary>
    /// <param name="Lanes"> Total number of lanes. </param>
    public sealed record Ramp(int Lanes) : Op
    {
        public static readonly ParameterInfo BaseOffset = new(typeof(Ramp), 0, "baseOffset");
        public static readonly ParameterInfo Stride = new(typeof(Ramp), 1, "stride");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType baseOffset, TensorType stride)
        {
            return new TensorType(baseOffset.DType, new[] { Lanes });
        }
    }
}