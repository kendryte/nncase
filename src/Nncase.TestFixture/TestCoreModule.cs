// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.TestFixture;

/// <summary>
/// K210 module.
/// </summary>
public class TestCoreModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<TestingProvider>().AsImplementedInterfaces().SingleInstance();
    }
}
