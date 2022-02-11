// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Evaluator;

/// <summary>
/// Evaluator module.
/// </summary>
public class EvaluatorModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<TypeInferenceProvider>().AsImplementedInterfaces();
        builder.RegisterType<EvaluateProvider>().AsImplementedInterfaces();
    }
}
