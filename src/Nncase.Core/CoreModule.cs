// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase;

/// <summary>
/// Evaluator module.
/// </summary>
public class CoreModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<CompilerServicesProvider>().AsImplementedInterfaces().SingleInstance();
    }
}
