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
/// QuantParam
/// </summary>
public struct QuantParam : IEquatable<QuantParam>
{
    /// <summary>
    /// Get the ZeroPoint
    /// </summary>
    public int ZeroPoint;
    
    /// <summary>
    /// Get the Scale
    /// </summary>
    public float Scale;

    public QuantParam(int zeroPoint, float scale)
    {
        ZeroPoint = zeroPoint;
        Scale = scale;
    }
    
    /// <inheritdoc/>
    public bool Equals(QuantParam other)
    {
        return Scale == other.Scale && ZeroPoint == other.ZeroPoint;
    }
}

/// <summary>
/// Prim type of <see cref="QuantizeParam"/>.
/// </summary>
public sealed record QuantParamType : PrimType
{
    /// <inheritdoc/>
    public override Type CLRType => typeof(QuantParam);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(QuantParam);

    public override PrimTypeAttributes Attributes { get; }

    public override string FullName => "QuantParam";
    public override string ShortName => "qp";
    public override TypeCode TypeCode => TypeCode.ValueType;
}