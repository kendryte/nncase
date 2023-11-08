// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Converters;

/// <summary>
/// Converters module.
/// </summary>
internal class ConvertersModule : IApplicationPart
{
    /// <inheritdoc/>
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BFloat16Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<BooleanConverters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<DoubleConverters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<HalfConverters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Int16Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Int32Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Int64Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<Int8Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<SingleConverters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UInt16Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UInt32Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UInt64Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UInt8Converters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PointerConverters>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<PointerIntConverters>(reuse: Reuse.Singleton);
    }
}
