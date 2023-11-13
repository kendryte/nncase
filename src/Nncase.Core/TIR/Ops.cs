﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR;

/// <summary>
/// Load op.
/// </summary>
public sealed partial class Load : Op
{
    /// <summary>
    /// Gets handle.
    /// </summary>
    public static readonly ParameterInfo Handle = new(typeof(Load), 0, "handle", IsPointer() | IsIntegralScalar());

    /// <summary>
    /// Gets index.
    /// </summary>
    public static readonly ParameterInfo Index = new(typeof(Load), 1, "index", IsIntegralScalar());

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}

/// <summary>
/// <see cref="T.Ramp(Expr, Expr, int)"/>.
/// </summary>
public sealed partial class Ramp : Op
{
    /// <summary>
    /// Gets offset.
    /// </summary>
    public static readonly ParameterInfo Offset = new(typeof(Ramp), 0, "offset", HasDataType(DataTypes.Int32) & IsScalar());

    /// <summary>
    /// Gets stride.
    /// </summary>
    public static readonly ParameterInfo Stride = new(typeof(Ramp), 1, "stride", HasDataType(DataTypes.Int32) & IsScalar());

    public int Lanes { get; }
}

/// <summary>
/// Store, return unit.
/// </summary>
public sealed partial class Store : Op
{
    /// <summary>
    /// The buffer variable handle.
    /// </summary>
    public static readonly ParameterInfo Handle = new(typeof(Store), 0, "handle", IsPointer() | IsIntegralScalar());

    /// <summary>
    /// The index locations to be stored.
    /// </summary>
    public static readonly ParameterInfo Index = new(typeof(Store), 1, "index", IsIntegralScalar());

    /// <summary>
    /// The value to be stored.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Store), 2, "value", IsScalar());

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}

/// <summary>
/// The Nop Expresstion, When We build the Ir, It's like the return the Void Value. We will skip it when print Ir/lower.
/// </summary>
public sealed partial class Nop : Op
{
    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
