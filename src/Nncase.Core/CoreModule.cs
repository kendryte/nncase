// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase;

/// <summary>
/// Core module.
/// </summary>
internal class CoreModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<CompilerServicesProvider>(reuse: Reuse.Singleton);
        registrator.Register<IDataTypeServiceProvider, DataTypeServiceProvider>(reuse: Reuse.Singleton);

        // Prim types
        registrator.Register<PrimType, BooleanType>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Utf8CharType>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Int8Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Int16Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Int32Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Int64Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, UInt8Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, UInt16Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, UInt32Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, UInt64Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Float16Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Float32Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, Float64Type>(reuse: Reuse.Singleton);
        registrator.Register<PrimType, BFloat16Type>(reuse: Reuse.Singleton);

        // Value types
        registrator.Register<ValueType, QuantParamType>(reuse: Reuse.Singleton);
    }
}
