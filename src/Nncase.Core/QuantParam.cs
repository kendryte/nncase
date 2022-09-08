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

    /// <summary>
    /// ctor
    /// </summary>
    /// <param name="zeroPoint"></param>
    /// <param name="scale"></param>
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

    /// <inheritdoc/>
    public override string ToString()
    {
        return $"<{ZeroPoint},{Scale}>";
    }
}

/// <summary>
/// Prim type of <see cref="QuantParam"/>.
/// </summary>
public sealed record QuantParamType : ValueType
{
    /// <inheritdoc/>
    public override Type CLRType => typeof(QuantParam);

    /// <inheritdoc/>
    public unsafe override int SizeInBytes => sizeof(QuantParam);

    /// <inheritdoc/>
    override public Guid Uuid { get; }

    /// <inheritdoc/>
    public override string ToString()
    {
        return "QuatParm";
    }
}