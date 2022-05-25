// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.Targets;

namespace Nncase;

/// <summary>
/// StackVM module.
/// </summary>
public class StackVMModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<CPUTarget>().AsImplementedInterfaces().SingleInstance();
    }
}
