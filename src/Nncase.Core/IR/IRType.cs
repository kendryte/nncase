// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Expression type.
    /// </summary>
    public abstract record IRType
    {
    }

    /// <summary>
    /// Any type.
    /// </summary>
    public sealed record AnyType : IRType
    {
        /// <summary>
        /// The default any type instance.
        /// </summary>
        public static readonly AnyType Default = new();

        private AnyType()
        {
        }
    }

    /// <summary>
    /// Invalid type.
    /// </summary>
    public sealed record InvalidType(string Reason) : IRType;

    /// <summary>
    /// Tensor type.
    /// </summary>
    public sealed record TensorType(DataType DataType, Shape Shape) : IRType
    {
        /// <summary>
        /// Gets a value indicating whether scalar.
        /// </summary>
        public bool IsScalar => Shape.IsScalar;

        /// <summary>
        /// Gets a value indicating whether tensor.
        /// </summary>
        public bool IsTensor => !IsScalar;

        /// <summary>
        /// Initialize a scalar tensor type.
        /// </summary>
        /// <param name="dataType">Data type.</param>
        /// <returns>The scalar tensor type.</returns>
        public static TensorType Scalar(DataType dataType) => new(dataType, Shape.Scalar);

        /// <summary>
        /// Initialize an unranked tensor type.
        /// </summary>
        /// <param name="dataType">Data type.</param>
        /// <returns>The unranked tensor type.</returns>
        public static TensorType Unranked(DataType dataType) => new(dataType, Shape.Unranked);

        /// <summary>
        /// Initialize an invalid tensor type.
        /// </summary>
        /// <param name="dataType">Data type.</param>
        /// <returns>The invalid tensor type.</returns>
        public static TensorType Invalid(DataType dataType) => new(dataType, Shape.Invalid);
    }

    /// <summary>
    /// Tuple type.
    /// </summary>
    public sealed record TupleType(ImmutableArray<IRType> Fields) : IRType
    {
        /// <summary>
        /// Void type.
        /// </summary>
        public static readonly TupleType Void = new(ImmutableArray<IRType>.Empty);
    }

    /// <summary>
    /// Callable type.
    /// </summary>
    public sealed record CallableType(IRType ReturnType, ImmutableArray<IRType> Parameters) : IRType;
}
