// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.K210;
using Nncase.Hosting;
using Nncase.Targets;

namespace Nncase;

/// <summary>
/// K210 module.
/// </summary>
internal sealed class K210Module : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<ITarget, K210Target>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FakeKPUConv2DEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FakeKPUDownloadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<FakeKPUUploadEvaluator>(reuse: Reuse.Singleton);
    }
}
