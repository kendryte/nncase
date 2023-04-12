// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;

namespace Nncase.Passes;

public sealed class EGraphConstructPass : Pass<BaseFunction, IEGraph>
{
    protected override Task<IEGraph> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        IEGraph post = new EGraph(input);
        return Task.FromResult(post);
    }

    protected override Task OnPassStartAsync(BaseFunction input, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            DumpScope.Current.DumpIR(input, "Start");
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
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
