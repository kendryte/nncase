// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.Evaluator.Math;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Random module.
/// </summary>
public class RandomModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<NormalEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<NormalLikeEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<UniformEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<UniformLikeEvaluator>().AsImplementedInterfaces();
    }
}
