// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase;

/// <summary>
/// Core module.
/// </summary>
public class CoreModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<CompilerServicesProvider>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<DataTypeServiceProvider>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<IR.IRPrinterProvider>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<CompileOptions>().AsImplementedInterfaces().SingleInstance();
            
        // Prim types
        builder.RegisterType<BooleanType>().As<PrimType>().SingleInstance();
        builder.RegisterType<Utf8CharType>().As<PrimType>().SingleInstance();
        builder.RegisterType<Int8Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<Int16Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<Int32Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<Int64Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<UInt8Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<UInt16Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<UInt32Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<UInt64Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<Float16Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<Float32Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<Float64Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<BFloat16Type>().As<PrimType>().SingleInstance();
        builder.RegisterType<QuantParamType>().As<PrimType>().SingleInstance();
    }
}
