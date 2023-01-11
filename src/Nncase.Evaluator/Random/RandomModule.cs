// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Evaluator.Math;
using Nncase.Hosting;

namespace Nncase.Evaluator.Random;

/// <summary>
/// Random module.
/// </summary>
internal class RandomModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<NormalEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<NormalLikeEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UniformEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<UniformLikeEvaluator>(reuse: Reuse.Singleton);
    }
}
