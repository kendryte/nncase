// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Converters;

internal class Vector32ToScalarConverter<TFrom, TTo> : ISpanConverter<Vector32<TFrom>, TTo>
    where TFrom : unmanaged, IEquatable<TFrom>
    where TTo : unmanaged, IEquatable<TTo>
{
    private readonly ISpanConverter<TFrom, TTo> _elementConverter;

    public Vector32ToScalarConverter(ISpanConverter<TFrom, TTo> elementConverter)
    {
        _elementConverter = elementConverter ?? throw new ArgumentNullException(nameof(elementConverter));
    }

    public void ConvertTo(ReadOnlySpan<Vector32<TFrom>> source, Span<TTo> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException("Cannot perform exact cast from vector to scalar type");
        }

        var elementsPerVector = Vector32<TFrom>.Count;
        var requiredDestSize = source.Length * elementsPerVector;

        if (dest.Length < requiredDestSize)
        {
            throw new ArgumentException("Destination buffer is not large enough for the flattened vector data");
        }

        var buffer = new TFrom[elementsPerVector];
        var bufferSpan = new Span<TFrom>(buffer);

        for (int i = 0; i < source.Length; i++)
        {
            var vector = source[i];

            for (int j = 0; j < elementsPerVector; j++)
            {
                buffer[j] = vector[j];
            }

            var destSlice = dest.Slice(i * elementsPerVector, elementsPerVector);
            _elementConverter.ConvertTo(bufferSpan, destSlice, castMode);
        }
    }
}

internal class ScalarToVector32Converter<TFrom, TTo> : ISpanConverter<TFrom, Vector32<TTo>>
    where TFrom : unmanaged, IEquatable<TFrom>
    where TTo : unmanaged, IEquatable<TTo>
{
    private readonly ISpanConverter<TFrom, TTo> _elementConverter;

    public ScalarToVector32Converter(ISpanConverter<TFrom, TTo> elementConverter)
    {
        _elementConverter = elementConverter ?? throw new ArgumentNullException(nameof(elementConverter));
    }

    public void ConvertTo(ReadOnlySpan<TFrom> source, Span<Vector32<TTo>> dest, CastMode castMode)
    {
        if (castMode == CastMode.Exact)
        {
            throw new InvalidCastException("Cannot perform exact cast from scalar to vector type");
        }

        var elementsPerVector = Vector32<TTo>.Count;
        var requiredSourceSize = dest.Length * elementsPerVector;

        if (source.Length < requiredSourceSize)
        {
            throw new ArgumentException("Source buffer does not contain enough elements to fill the vectors");
        }

        var buffer = new TTo[elementsPerVector];
        var bufferSpan = new Span<TTo>(buffer);

        for (int i = 0; i < dest.Length; i++)
        {
            var sourceSlice = source.Slice(i * elementsPerVector, elementsPerVector);
            _elementConverter.ConvertTo(sourceSlice, bufferSpan, castMode);

            var vector = default(Vector32<TTo>);
            for (int j = 0; j < elementsPerVector; j++)
            {
                vector[j] = buffer[j];
            }

            dest[i] = vector;
        }
    }
}

internal class Vector32Converter<TFrom, TTo> : ISpanConverter<Vector32<TFrom>, Vector32<TTo>>
    where TTo : unmanaged, IEquatable<TTo>
    where TFrom : unmanaged, IEquatable<TFrom>
{
    private readonly ISpanConverter<TFrom, TTo> _elementConverter;

    public Vector32Converter(ISpanConverter<TFrom, TTo> elementConverter)
    {
        _elementConverter = elementConverter ?? throw new ArgumentNullException(nameof(elementConverter));
    }

    public void ConvertTo(ReadOnlySpan<Vector32<TFrom>> source, Span<Vector32<TTo>> dest, CastMode castMode)
    {
        if (dest.Length < source.Length)
        {
            throw new ArgumentException("Destination buffer is not sufficient.");
        }

        for (int i = 0; i < source.Length; i++)
        {
            _elementConverter.ConvertTo(source[i].AsSpan(), dest[i].AsSpan(), castMode);
        }
    }
}
