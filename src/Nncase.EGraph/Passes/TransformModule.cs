// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Passes;

/// <summary>
/// Transform module.
/// </summary>
internal class TransformModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<IEGraphExtractor, EGraphExtractors.SatExtractor>(reuse: Reuse.ScopedOrSingleton);
        registrator.Register<IEGraphExtractor, EGraphExtractors.SatExtractor>(reuse: Reuse.ScopedOrSingleton, made: Parameters.Of.Type<Evaluator.ICostEvaluateProvider>(serviceKey: Evaluator.CostEvaluatorKinds.Online), serviceKey: Evaluator.CostEvaluatorKinds.Online);
        registrator.Register<IEGraphRewriteProvider, EGraphRewriteProvider>(reuse: Reuse.ScopedOrSingleton);
    }
}
