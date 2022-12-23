// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.PatternMatch;

/// <summary>
/// PatternMatch module.
/// </summary>
public class PatternMatchModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<EGraphMatchProvider>().AsImplementedInterfaces().SingleInstance();
    }
}
