// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// TIR module.
/// </summary>
public class TIRModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<LoadEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<RampEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<StoreEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<NopEvaluator>().AsImplementedInterfaces();
    }
}
