// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.Evaluator.K210;
using Nncase.Targets;

namespace Nncase;

/// <summary>
/// K210 module.
/// </summary>
public class K210Module : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<K210Target>().AsImplementedInterfaces().SingleInstance();
        builder.RegisterType<FakeKPUConv2DEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<FakeKPUDownloadEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<FakeKPUUploadEvaluator>().AsImplementedInterfaces();
    }
}
