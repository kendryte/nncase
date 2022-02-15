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
/// <typeparam name="TFrom">From type.</typeparam>
/// <typeparam name="TTo">To type.</typeparam>
public interface ISpanConverter<TFrom, TTo>
    where TFrom : unmanaged, IEquatable<TFrom>
    where TTo : unmanaged, IEquatable<TTo>
{
    /// <summary>
    /// Convert span.
    /// </summary>
    /// <param name="source">Source span.</param>
    /// <param name="dest">Dest span.</param>
    void ConvertTo(ReadOnlySpan<TFrom> source, Span<TTo> dest);
}
