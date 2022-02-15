// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

internal interface IDataTypeServiceProvider
{
    PrimType GetPrimTypeFromType(Type type);
}

internal class DataTypeServiceProvider : IDataTypeServiceProvider
{
    private readonly Dictionary<Type, PrimType> _primTypes = new();

    public DataTypeServiceProvider(PrimType[] primTypes)
    {
        _primTypes = primTypes.ToDictionary(x => x.CLRType);
    }

    public PrimType GetPrimTypeFromType(Type type)
    {
        return _primTypes[type];
    }
}
