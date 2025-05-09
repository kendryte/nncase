// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TypeCode = Nncase.Runtime.TypeCode;

namespace Nncase;

/// <summary>
/// Prim type of <see cref="Memory{T}"/>.
/// </summary>
public sealed record MemoryType(DataType ElemType) : ValueType
{
    /// <inheritdoc/>
    public override Type CLRType => typeof(Memory<>).MakeGenericType(ElemType.CLRType);

    /// <inheritdoc/>
    public override int SizeInBytes => 1;

    /// <inheritdoc/>
    public override Guid Uuid { get; } = Guid.Parse("47020b25-b497-4ec7-b781-e1f435926c42");

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"Memory<{ElemType}>";
    }
}
