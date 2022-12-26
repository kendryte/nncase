﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// Expression.
/// </summary>
public abstract partial record Expr
{
    /// <summary>
    /// hash code cache.
    /// </summary>
    protected int? _hashcode;

    /// <summary>
    /// checked type.
    /// </summary>
    public IRType? CheckedType;

    /// <summary>
    /// Gets checked shape.
    /// </summary>
    public Shape CheckedShape => (CheckedType ?? ((Const)this).ValueType) switch
    {
        TensorType type => type.Shape,
        _ => throw new InvalidOperationException("Only The Expr Have CheckedType Can Get It's Shape"),
    };

    /// <summary>
    /// Gets if this expr is tensortype, can return the checkedDatatype.
    /// </summary>
    public DataType CheckedDataType => CheckedType switch
    {
        // todo:more info
        TensorType type => type.DType,
        _ => throw new InvalidOperationException("Expr don't have a valid tensor type"),
    };

    /// <inheritdoc/>
    public virtual bool Equals(Expr? other)
    {
        return !(other is null) && EqualityContract == other.EqualityContract;
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return _hashcode ??= EqualityComparer<Type>.Default.GetHashCode(EqualityContract);
    }

    protected virtual bool PrintMembers(System.Text.StringBuilder builder)
    {
        return false;
    }
}
