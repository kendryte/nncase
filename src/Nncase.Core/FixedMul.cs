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
public record struct FixedMul(float Mul, sbyte Shift)
{
    /// <summary>
    /// Gets get rounded mul.
    /// </summary>
    public int RoundedMul => (int)Math.Round(Mul);
}
