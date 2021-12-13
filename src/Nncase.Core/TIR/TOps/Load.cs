using Nncase.IR;
using System;
namespace Nncase.TIR
{
    /// <summary>
    /// <seealso cref="F.TOps.Load(Var, Expr, Expr?)"/>
    /// </summary>
    public sealed record Load() : Op
    {
        /// <summary>
        /// The pointer variable in the load expression.
        /// </summary>
        public static readonly ParameterInfo BufferHandle = new(typeof(Load), 0, "bufferHandle");

        /// <summary>
        /// The index in the load.
        /// </summary>
        public static readonly ParameterInfo Index = new(typeof(Load), 1, "index");

        /// <summary>
        /// The load predicate.
        /// </summary>
        public static readonly ParameterInfo Predicate = new(typeof(Load), 2, "predicate");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, PointerType bufferHandle, TensorType index, TensorType predicate)
        {
            return TensorType.Scalar(bufferHandle.DType);
        }
    }

}