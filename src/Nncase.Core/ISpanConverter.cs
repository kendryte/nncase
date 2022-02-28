// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Span converter.
/// </summary>
public interface ISpanConverter
{
}

/// <summary>
/// Span converter.
/// </summary>
/// <typeparam name="TFrom">From type.</typeparam>
/// <typeparam name="TTo">To type.</typeparam>
public interface ISpanConverter<TFrom, TTo> : ISpanConverter
    where TFrom : unmanaged, IEquatable<TFrom>
    where TTo : unmanaged, IEquatable<TTo>
{
    /// <summary>
    /// Convert span.
    /// </summary>
    /// <param name="source">Source span.</param>
    /// <param name="dest">Dest span.</param>
    /// <param name="castMode">Cast mode.</param>
    void ConvertTo(ReadOnlySpan<TFrom> source, Span<TTo> dest, CastMode castMode);
}

/// <summary>
/// Pointer span converter.
/// </summary>
/// <typeparam name="TTo">To type.</typeparam>
public interface IPointerSpanConverter<TTo> : ISpanConverter
    where TTo : unmanaged, IEquatable<TTo>
{
    /// <summary>
    /// Convert span.
    /// </summary>
    /// <typeparam name="T">Pointer elem type.</typeparam>
    /// <param name="source">Source span.</param>
    /// <param name="dest">Dest span.</param>
    /// <param name="castMode">Cast mode.</param>
    void ConvertTo<T>(ReadOnlySpan<Pointer<T>> source, Span<TTo> dest, CastMode castMode)
        where T : unmanaged, IEquatable<T>;
}
