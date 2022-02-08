// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Autofac;

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Imaging module.
/// </summary>
public class ImagingModule : Module
{
    /// <inheritdoc/>
    protected override void Load(ContainerBuilder builder)
    {
        builder.RegisterType<ResizeImageEvaluator>().AsImplementedInterfaces();
    }
}
