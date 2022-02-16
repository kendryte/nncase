// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Converters;

/// <summary>
/// Converters module.
/// </summary>
public class ConvertersModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<BFloat16Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<BooleanConverters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<DoubleConverters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<HalfConverters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<Int16Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<Int32Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<Int64Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<Int8Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<SingleConverters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<UInt16Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<UInt32Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<UInt64Converters>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<UInt8Converters>().AsImplementedInterfaces().SingleInstance();
    }
}
