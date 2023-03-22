// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;
using Nncase.PatternMatch;

namespace Nncase.Passes;

/// <summary>
/// Transform module.
/// </summary>
internal class TransformModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<RewriteProvider>(reuse: Reuse.Singleton);
    }
}
