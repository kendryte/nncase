// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Diagnostics;
using Nncase.Hosting;
using Nncase.IR;

namespace Nncase.Diagnostics;

/// <summary>
/// Diagnostics module.
/// </summary>
internal class DiagnosticsModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.Register<IIRPrinterProvider, IRPrinterProvider>(reuse: Reuse.Singleton);

        registrator.Register<IDumpperFactory, DumpperFactory>(reuse: Reuse.Scoped);
    }
}
