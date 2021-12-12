using Nncase.IR;
using System;
namespace Nncase.TIR
{
    /// <summary>
    /// <seealso cref="F.TOps.Load(DataType, Var, Expr, Expr?)"/>
    /// </summary>
    /// <param name="LoadType"> Load value's DataType </param>
    public sealed record Load(DataType LoadType) : Op
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
            if (predicate.DType != LoadType)
            {
                return new InvalidType($"The Predicate {predicate.DType} != LoadType {LoadType}");
            }
            if (!LoadType.CompatibleWith(bufferHandle.DType))
            {
                return new InvalidType($"The LoadType {predicate.DType} not Compatible With bufferHandle {bufferHandle.DType}");
            }
            return TensorType.Scalar(LoadType);
        }
    }

}