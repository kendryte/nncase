// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
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
    public sealed record TensorType(DataType DType, Shape Shape) : IRType
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
        /// <param name="DType">Data type.</param>
        /// <returns>The scalar tensor type.</returns>
        public static TensorType Scalar(DataType DType) => new(DType, Shape.Scalar);

        /// <summary>
        /// Initialize an unranked tensor type.
        /// </summary>
        /// <param name="DType">Data type.</param>
        /// <returns>The unranked tensor type.</returns>
        public static TensorType Unranked(DataType DType) => new(DType, Shape.Unranked);

        /// <summary>
        /// Initialize an invalid tensor type.
        /// </summary>
        /// <param name="DType">Data type.</param>
        /// <returns>The invalid tensor type.</returns>
        public static TensorType Invalid(DataType DType) => new(DType, Shape.Invalid);
    }

    /// <summary>
    /// Tuple type.
    /// </summary>
    public sealed record TupleType(IRArray<IRType> Fields) : IRType, IEnumerable<IRType>, IReadOnlyList<IRType>
    {
        /// <summary>
        /// Void type.
        /// </summary>
        public static readonly TupleType Void = new(ImmutableArray<IRType>.Empty);

        public TupleType(IEnumerable<IRType> Fields) : this(Fields.ToImmutableArray()) { }

        public IRType this[int index] => ((IReadOnlyList<IRType>)Fields)[index];

        public int Count => ((IReadOnlyCollection<IRType>)Fields).Count;

        public IEnumerator<IRType> GetEnumerator()
        {
            return ((IEnumerable<IRType>)Fields).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)Fields).GetEnumerator();
        }
    }

    /// <summary>
    /// Callable type.
    /// </summary>
    public sealed record CallableType(IRType ReturnType, IRArray<IRType> Parameters) : IRType;


    /// <summary>
    /// Low-level raw pointer type.
    /// PointerType represents type hints in the TIR to be
    /// passed to the final code generator.
    /// PointerType should not occur in the high-level analysis.
    /// </summary>
    /// <param name="DType">The type of the element which the pointer points to.</param>
    /// <param name="StorageScope">The storage scope of the pointer</param>
    public sealed record PointerType(DataType DType, string StorageScope) : IRType
    { }
}
