// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.IR.Distributed;

/// <summary>
/// Distributed module.
/// </summary>
internal class DistributedModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<BoxingEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<ForceBoxingEvaluator>(reuse: Reuse.Singleton);
    }
}
