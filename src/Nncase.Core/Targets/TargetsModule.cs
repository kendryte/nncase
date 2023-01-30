// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Targets;

/// <summary>
/// Targets module.
/// </summary>
internal class TargetsModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<ITargetProvider, TargetProvider>(reuse: Reuse.Singleton);
    }
}
