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
/// fixed mul.
/// </summary>
public struct FixedMul : IEquatable<FixedMul>
{
    /// <summary>
    /// mul params.
    /// </summary>
    public float Mul;

    /// <summary>
    /// shift params.
    /// </summary>
    public sbyte Shift;

    /// <summary>
    /// Initializes a new instance of the <see cref="FixedMul"/> struct.
    /// ctor.
    /// </summary>
    /// <param name="mul"></param>
    /// <param name="shift"></param>
    public FixedMul(float mul, sbyte shift)
    {
        Mul = mul;
        Shift = shift;
    }

    /// <summary>
    /// Gets get rounded mul.
    /// </summary>
    public int RoundedMul => (int)Math.Round(Mul);

    /// <inheritdoc/>
    public bool Equals(FixedMul other)
    {
        return Mul == other.Mul && Shift == other.Shift;
    }

    public override bool Equals(object obj)
    {
        return obj is FixedMul && Equals((FixedMul)obj);
    }
}

