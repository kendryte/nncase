// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Targets;

/// <summary>
/// Targets module.
/// </summary>
public class TargetsModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<TargetProvider>().AsImplementedInterfaces().SingleInstance();
    }
}
