// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;
using Nncase.Evaluator.Tensors;

namespace Nncase.Evaluator.Buffer;

/// <summary>
/// Buffer module.
/// </summary>
public class BufferModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<DDrOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<BaseMentOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<StrideOfEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<AllocateEvaluator>().AsImplementedInterfaces();
        builder.RegisterType<UninitializedEvaluator>().AsImplementedInterfaces();
    }
}
