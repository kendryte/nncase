// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Imaging;

/// <summary>
/// Imaging module.
/// </summary>
internal class ImagingModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<ResizeImageEvaluator>(reuse: Reuse.Singleton);
    }
}
