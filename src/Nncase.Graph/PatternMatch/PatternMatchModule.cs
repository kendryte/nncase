// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using DryIoc;
using Nncase.Hosting;

namespace Nncase.PatternMatch;

/// <summary>
/// PatternMatch module.
/// </summary>
internal class PatternMatchModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<MatchProvider>(reuse: Reuse.Singleton);
    }
}
