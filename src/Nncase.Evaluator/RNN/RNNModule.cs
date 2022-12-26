// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.Evaluator.NN;

namespace Nncase.Evaluator.RNN;

/// <summary>
/// RNN module.
/// </summary>
public class RNNModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<LSTMEvaluator>().AsImplementedInterfaces();
    }
}
