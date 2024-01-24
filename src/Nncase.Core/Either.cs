// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

internal enum EitherState
{
    None,
    T1,
    T2,
}

public struct Either<T1, T2>
{
    private readonly EitherState _state;

    [AllowNull]
    private readonly T1 _t1;

    [AllowNull]
    private readonly T2 _t2;

    private Either(T1 value)
    {
        _state = EitherState.T1;
        _t1 = value;
        _t2 = default;
    }

    private Either(T2 value)
    {
        _state = EitherState.T2;
        _t2 = value;
        _t1 = default;
    }

    public T1 Value1 => _state == EitherState.T1 ? _t1 : throw new InvalidOperationException($"Value is not a {typeof(T1)}.");

    public T2 Value2 => _state == EitherState.T2 ? _t2 : throw new InvalidOperationException($"Value is not a {typeof(T2)}.");

    public static implicit operator Either<T1, T2>(T1 value) => new(value);

    public static implicit operator Either<T1, T2>(T2 value) => new(value);

    public static explicit operator T1(Either<T1, T2> value) => value.Value1;

    public static explicit operator T2(Either<T1, T2> value) => value.Value2;

    public static Either<T1, T2> From(T1 value) => new(value);

    public static Either<T1, T2> From(T2 value) => new(value);

    public static Either<T1, T2> From<TCommon>(TCommon value)
        => value switch
        {
            T1 v => v,
            T2 v => v,
            _ => throw new InvalidCastException(),
        };

    public bool Is<T>()
    {
        if (typeof(T) == typeof(T1))
        {
            return _state == EitherState.T1;
        }

        if (typeof(T) == typeof(T2))
        {
            return _state == EitherState.T2;
        }

        return false;
    }

    public T Match<T>(Func<T1, T> t1Selector, Func<T2, T> t2Selector)
        => Is<T1>() ? t1Selector(_t1) : t2Selector(_t2);

    public object Match(Func<T1, object> t1Selector, Func<T2, object> t2Selector)
        => Is<T1>() ? t1Selector(_t1) : t2Selector(_t2);

    public TCommon ToCommonValue<TCommon>()
        where TCommon : class
    {
        return Is<T1>() ? (TCommon)(object)Value1! : (TCommon)(object)Value2!;
    }
}
