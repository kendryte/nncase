// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using static Nncase.IR.Utility;

namespace Nncase.IR.Tensors
{
    public sealed record RandomNormal(DataType Type) : Op
    {
        /// <summary>
        /// Gets mean.
        /// </summary>
        public static readonly ParameterInfo Mean = new(typeof(RandomNormal), 0, "mean", IsFloatScalar());

        /// <summary>
        /// Gets scale.
        /// </summary>
        public static readonly ParameterInfo Scale = new(typeof(RandomNormal), 1, "scale", IsFloatScalar());

        /// <summary>
        /// Gets seed.
        /// </summary>
        public static readonly ParameterInfo Seed = new(typeof(RandomNormal), 2, "seed", IsFloatScalar());
        
        /// <summary>
        /// Gets shape.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(RandomNormal), 3, "shape", IsIntegral() & HasRank(1));
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType mean, TensorType scale, TensorType seed, TensorType shape)
        {
            if (context.GetArgument(this, Shape) is Const shapeValue)
            {
                return new TensorType(Type, new Shape(shapeValue.ToArray<int>()));
            }
            else
            {
                return new TensorType(Type, IR.Shape.Unranked);
            }
        }
    }
    
    public sealed record RandomNormalLike(DataType Type) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Slice), 0, "input");

        /// <summary>
        /// Gets mean.
        /// </summary>
        public static readonly ParameterInfo Mean = new(typeof(RandomNormalLike), 1, "mean", IsFloatScalar());

        /// <summary>
        /// Gets scale.
        /// </summary>
        public static readonly ParameterInfo Scale = new(typeof(RandomNormalLike), 2, "scale", IsFloatScalar());

        /// <summary>
        /// Gets seed.
        /// </summary>
        public static readonly ParameterInfo Seed = new(typeof(RandomNormalLike), 3, "seed", IsFloatScalar());
        
        /// <summary>
        /// Gets shape.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(RandomNormalLike), 4, "shape", IsIntegral() & HasRank(1));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType mean,
            TensorType scale, TensorType seed, TensorType shape) => input with {DType = Type};
    }

    public sealed record RandomUniform(DataType Type) : Op
    {
        /// <summary>
        /// Gets high.
        /// </summary>
        public static readonly ParameterInfo High = new(typeof(RandomUniform), 0, "high", IsFloatScalar());

        /// <summary>
        /// Gets low.
        /// </summary>
        public static readonly ParameterInfo Low = new(typeof(RandomUniform), 1, "low", IsFloatScalar());

        /// <summary>
        /// Gets seed.
        /// </summary>
        public static readonly ParameterInfo Seed = new(typeof(RandomUniform), 2, "seed", IsFloatScalar());
        
        /// <summary>
        /// Gets shape.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(RandomUniform), 3, "shape", IsIntegral() & HasRank(1));
        
        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType high, TensorType low, TensorType seed, TensorType shape)
        {
            if (context.GetArgument(this, Shape) is Const shapeValue)
            {
                return new TensorType(Type, new Shape(shapeValue.ToArray<int>()));
            }
            else
            {
                return new TensorType(Type, IR.Shape.Unranked);
            }
        }
    }

    public sealed record RandomUniformLike(DataType Type) : Op
    {
        /// <summary>
        /// Gets input.
        /// </summary>
        public static readonly ParameterInfo Input = new(typeof(Slice), 0, "input");
        
        /// <summary>
        /// Gets high.
        /// </summary>
        public static readonly ParameterInfo High = new(typeof(RandomUniform), 1, "high", IsFloatScalar());

        /// <summary>
        /// Gets low.
        /// </summary>
        public static readonly ParameterInfo Low = new(typeof(RandomUniform), 2, "low", IsFloatScalar());

        /// <summary>
        /// Gets seed.
        /// </summary>
        public static readonly ParameterInfo Seed = new(typeof(RandomUniform), 3, "seed", IsFloatScalar());
        
        /// <summary>
        /// Gets shape.
        /// </summary>
        public static readonly ParameterInfo Shape = new(typeof(RandomUniform), 4, "shape", IsIntegral() & HasRank(1));

        /// <inheritdoc/>
        public IRType InferInvokeResultType(ITypeInferenceContext context, TensorType input, TensorType high,
            TensorType low, TensorType seed, TensorType shape) => input with { DType = Type };
    }

}