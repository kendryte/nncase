using System;
using static Nncase.IR.Utility;

namespace Nncase.IR.NN
{
    public sealed record L2Normalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input) => input;
    }
    
    public sealed record BatchNormalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");

        /// <summary>
        /// Gets Epsilon.
        /// </summary>
        public static readonly ParameterInfo Epsilon = new(typeof(L2Normalization), 1, "epsilon", IsFloatScalar());

        /// <summary>
        /// Gets Momentum.
        /// </summary>
        public static readonly ParameterInfo Momentum = new(typeof(L2Normalization), 2, "momentum", IsFloatScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType epsilon, TensorType momentum) => input;
    }
    
    public sealed record InstanceNormalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");

        /// <summary>
        /// Gets Epsilon.
        /// </summary>
        public static readonly ParameterInfo Epsilon = new(typeof(L2Normalization), 1, "epsilon", IsFloatScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType epsilon) => input;
    }
    
    public sealed record LpNormalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");

        /// <summary>
        /// Gets Axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(L2Normalization), 1, "axis", IsIntegralScalar());
        
        /// <summary>
        /// Gets P.
        /// </summary>
        public static readonly ParameterInfo P = new(typeof(L2Normalization), 2, "p", IsFloatScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis, TensorType p) => input;
    }
    
    public sealed record LRN() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(L2Normalization), 0, "input");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Alpha = new(typeof(L2Normalization), 1, "alpha", IsFloatScalar());
        
        /// <summary>
        /// Gets beta.
        /// </summary>
        public static readonly ParameterInfo Beta = new(typeof(L2Normalization), 2, "beta", IsFloatScalar());

        /// <summary>
        /// Gets bias.
        /// </summary>
        public static readonly ParameterInfo Bias = new(typeof(L2Normalization), 3, "bias", IsFloatScalar());
        
        /// <summary>
        /// Gets size.
        /// </summary>
        public static readonly ParameterInfo Size = new(typeof(L2Normalization), 4, "size", IsIntegralScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType axis, TensorType p) => input;
    }
}