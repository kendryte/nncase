// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;
using Nncase.Passes.Analysis;

namespace Nncase.Passes;

/// <summary>
/// Analysis module.
/// </summary>
internal class AnalysisModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<IAnalyzerFactory, ExprUserAnalyzerFactory>(reuse: Reuse.Singleton);
    }
}
