// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.Utility;

namespace Nncase.Pattern;

public sealed record TensorConstPattern(Func<TensorConst, bool> Cond) : ExprPattern
{
    /// <summary>
    /// <see cref="Target"/>.
    /// </summary>
    private readonly TensorConst? _target = null;

    /// <summary>
    /// save the target const for match, we can print it for debug.
    /// </summary>
    public TensorConst? Target { get => _target; }

    public TensorConstPattern(TensorConst expr) : this(x => x == expr)
    {
        _target = expr;
    }

    public static implicit operator TensorConstPattern(byte value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(ushort value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(uint value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(ulong value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(sbyte value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(short value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(int value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(long value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(Half value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(float value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(double value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(BFloat16 value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(bool value) => new TensorConstPattern((TensorConst)value);

    public static implicit operator TensorConstPattern(Tensor value) => new TensorConstPattern((TensorConst)value);

    public bool MatchLeaf(TensorConst expr)
    {
        return Cond(expr) && MatchCheckedType(expr);
    }
}

public static partial class Utility
{
    public static TensorConstPattern IsTensorConst() => new TensorConstPattern(x => x is TensorConst);

    public static TensorConstPattern IsTensorConst(Func<TensorConst, bool> Cond) => new TensorConstPattern(Cond);

    public static TensorConstPattern IsTensorConst(TypePattern typePattern) => new TensorConstPattern(x => typePattern.MatchLeaf(x.ValueType));
}
