// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;

namespace Nncase.Passes;

/// <summary>
/// EGraph pass.
/// </summary>
public abstract class EGraphPass : Pass<IEGraph, IEGraph>
{
    protected override Task OnPassStartAsync(IEGraph input, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            using var fs = DumpScope.Current.OpenFile(Path.Combine("Start", $"V{input.Version}.dot"));
            EGraphPrinter.DumpEgraphAsDot(input, null, fs);
        }

        return Task.CompletedTask;
    }

    protected override Task OnPassEndAsync(IEGraph post, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            using var fs = DumpScope.Current.OpenFile(Path.Combine("End", $"V{post.Version}.dot"));
            EGraphPrinter.DumpEgraphAsDot(post, null, fs);
        }

        return Task.CompletedTask;
    }
}
