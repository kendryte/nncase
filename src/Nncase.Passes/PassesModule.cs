// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Passes;

/// <summary>
/// Passes module.
/// </summary>
internal class PassesModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<IPassManagerFactory, PassManagerFactory>(reuse: Reuse.Singleton);
        registrator.Register<IAnalyzerManager, AnalyzerManager>(reuse: Reuse.Singleton);
    }
}
