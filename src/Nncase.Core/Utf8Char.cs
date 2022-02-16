// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

/// <summary>
/// Utf8 char.
/// </summary>
public struct Utf8Char
{
    private byte _value;

    /// <summary>
    /// Implicit convert <see cref="Utf8Char"/> to <see cref="byte"/>.
    /// </summary>
    /// <param name="char">Utf8 char.</param>
    public static implicit operator byte(Utf8Char @char) => @char._value;

    /// <summary>
    /// Implicit convert <see cref="byte"/> to <see cref="Utf8Char"/>.
    /// </summary>
    /// <param name="byte">Byte.</param>
    public static implicit operator Utf8Char(byte @byte)
    {
        Utf8Char @char;
        Unsafe.SkipInit(out @char);
        @char._value = @byte;
        return @char;
    }
}
