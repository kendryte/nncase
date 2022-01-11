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
        public static readonly ParameterInfo Input = new(typeof(BatchNormalization), 0, "input");

        /// <summary>
        /// Gets scale.
        /// </summary>
        public static readonly ParameterInfo Scale = new(typeof(BatchNormalization), 1, "scale", IsFloatScalar());

        /// <summary>
        /// Gets bias.
        /// </summary>
        public static readonly ParameterInfo Bias = new(typeof(BatchNormalization), 2, "bias", IsFloatScalar());
        
        /// <summary>
        /// Gets input_mean.
        /// </summary>
        public static readonly ParameterInfo InputMean = new(typeof(BatchNormalization), 3, "input_mean", IsFloatScalar());

        /// <summary>
        /// Gets input_var.
        /// </summary>
        public static readonly ParameterInfo InputVar = new(typeof(BatchNormalization), 4, "input_var", IsFloatScalar());

        /// <summary>
        /// Gets epsilon.
        /// </summary>
        public static readonly ParameterInfo Epsilon = new(typeof(BatchNormalization), 5, "epsilon", IsFloatScalar());

        /// <summary>
        /// Gets momentum.
        /// </summary>
        public static readonly ParameterInfo Momentum = new(typeof(BatchNormalization), 6, "momentum", IsFloatScalar());
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType scale, TensorType bias, 
            TensorType input_mean, TensorType input_var, TensorType epsilon, TensorType momentum) => input;
    }
    
    public sealed record InstanceNormalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(InstanceNormalization), 0, "input");

        /// <summary>
        /// Gets Epsilon.
        /// </summary>
        public static readonly ParameterInfo Epsilon = new(typeof(InstanceNormalization), 1, "epsilon", IsFloatScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType epsilon) => input;
    }
    
    public sealed record LpNormalization() : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(LpNormalization), 0, "input");

        /// <summary>
        /// Gets Axis.
        /// </summary>
        public static readonly ParameterInfo Axis = new(typeof(LpNormalization), 1, "axis", IsIntegralScalar());
        
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
        public static readonly ParameterInfo Input = new(typeof(LRN), 0, "input");

        /// <summary>
        /// Gets axis.
        /// </summary>
        public static readonly ParameterInfo Alpha = new(typeof(LRN), 1, "alpha", IsFloatScalar());
        
        /// <summary>
        /// Gets beta.
        /// </summary>
        public static readonly ParameterInfo Beta = new(typeof(LRN), 2, "beta", IsFloatScalar());

        /// <summary>
        /// Gets bias.
        /// </summary>
        public static readonly ParameterInfo Bias = new(typeof(LRN), 3, "bias", IsFloatScalar());
        
        /// <summary>
        /// Gets size.
        /// </summary>
        public static readonly ParameterInfo Size = new(typeof(LRN), 4, "size", IsIntegralScalar());

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, 
            TensorType alpha, TensorType beta, TensorType bias, TensorType size) => input;
    }
}