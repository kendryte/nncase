// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// TIR module.
/// </summary>
internal class TIRModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<LoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<RampEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StoreEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NopEvaluator>(reuse: Reuse.Singleton);
    }
}
