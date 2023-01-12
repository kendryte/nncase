// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc;

namespace Nncase;

internal interface IDataTypeServiceProvider
{
    PrimType GetPrimTypeFromType(Type type);

    PrimType GetPrimTypeFromTypeCode(Runtime.TypeCode typeCode);

    ValueType GetValueTypeFromType(Type type);

    DataType GetDataTypeFromType(Type type);

    ISpanConverter GetConverter(Type fromType, Type toType);
}

internal class DataTypeServiceProvider : IDataTypeServiceProvider
{
    private readonly Dictionary<RuntimeTypeHandle, PrimType> _primTypes = new();
    private readonly Dictionary<Runtime.TypeCode, PrimType> _typeCodeToPrimTypes = new();
    private readonly Dictionary<RuntimeTypeHandle, ValueType> _valueTypes = new();
    private readonly IResolver _resolver;

    public DataTypeServiceProvider(PrimType[] primTypes, ValueType[] valueTypes, IResolver resolver)
    {
        _primTypes = primTypes.ToDictionary(x => x.CLRType.TypeHandle);
        _typeCodeToPrimTypes = primTypes.Where(x => x.TypeCode < Runtime.TypeCode.ValueType).ToDictionary(x => x.TypeCode);
        _valueTypes = valueTypes.ToDictionary(x => x.CLRType.TypeHandle);
        _resolver = resolver;
    }

    public ISpanConverter GetConverter(Type fromType, Type toType)
    {
        if (fromType.IsGenericType && fromType.GetGenericTypeDefinition() == typeof(Pointer<>))
        {
            var converter = _resolver.Resolve(typeof(IPointerSpanConverter<>).MakeGenericType(toType));
            var wrapperType = typeof(PointerSpanConverter<,>).MakeGenericType(fromType.GenericTypeArguments[0], toType);
            return (ISpanConverter)Activator.CreateInstance(wrapperType, converter)!;
        }
        else
        {
            return (ISpanConverter)_resolver.Resolve(typeof(ISpanConverter<,>).MakeGenericType(fromType, toType));
        }
    }

    public DataType GetDataTypeFromType(Type type)
    {
        if (_primTypes.TryGetValue(type.TypeHandle, out var primType))
        {
            return primType;
        }
        else if (_valueTypes.TryGetValue(type.TypeHandle, out var valueType))
        {
            return valueType;
        }

        throw new NotSupportedException($"Unsupported Type {type} in GetDataTypefromType");
    }

    public PrimType GetPrimTypeFromType(Type type)
    {
        return _primTypes[type.TypeHandle];
    }

    public PrimType GetPrimTypeFromTypeCode(Runtime.TypeCode typeCode)
    {
        return _typeCodeToPrimTypes[typeCode];
    }

    public ValueType GetValueTypeFromType(Type type)
    {
        return _valueTypes[type.TypeHandle];
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
