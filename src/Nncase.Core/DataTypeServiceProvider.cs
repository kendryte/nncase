// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Autofac;

namespace Nncase;

internal interface IDataTypeServiceProvider
{
    PrimType GetPrimTypeFromType(Type type);

    ISpanConverter GetConverter(Type fromType, Type toType);
}

internal class DataTypeServiceProvider : IDataTypeServiceProvider
{
    private readonly Dictionary<Type, PrimType> _primTypes = new();
    private readonly IComponentContext _componentContext;

    public DataTypeServiceProvider(PrimType[] primTypes, IComponentContext componentContext)
    {
        _primTypes = primTypes.ToDictionary(x => x.CLRType);
        _componentContext = componentContext;
    }

    public ISpanConverter GetConverter(Type fromType, Type toType)
    {
        if (fromType.IsGenericType && fromType.GetGenericTypeDefinition() == typeof(Pointer<>))
        {
            var converter = _componentContext.Resolve(typeof(IPointerSpanConverter<>).MakeGenericType(toType));
            var wrapperType = typeof(PointerSpanConverter<,>).MakeGenericType(fromType.GenericTypeArguments[0], toType);
            return (ISpanConverter)Activator.CreateInstance(wrapperType, converter)!;
        }
        else
        {
            return (ISpanConverter)_componentContext.Resolve(typeof(ISpanConverter<,>).MakeGenericType(fromType, toType));
        }
    }

    public PrimType GetPrimTypeFromType(Type type)
    {
        return _primTypes[type];
    }

    private class PointerSpanConverter<TElem, TTo> : ISpanConverter<Pointer<TElem>, TTo>
        where TElem : unmanaged, IEquatable<TElem>
        where TTo : unmanaged, IEquatable<TTo>
    {
        private readonly IPointerSpanConverter<TTo> _spanConverter;

        public PointerSpanConverter(IPointerSpanConverter<TTo> spanConverter)
        {
            _spanConverter = spanConverter;
        }

        public void ConvertTo(ReadOnlySpan<Pointer<TElem>> source, Span<TTo> dest, CastMode castMode)
        {
            _spanConverter.ConvertTo(source, dest, castMode);
        }
    }
}
