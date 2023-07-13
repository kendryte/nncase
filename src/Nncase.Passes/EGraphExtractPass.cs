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

public sealed class EGraphExtractPass : Pass<IEGraph, BaseFunction>
{
    private readonly IEGraphExtractor _extractor;

    public EGraphExtractPass(IEGraphExtractor extractor)
    {
        _extractor = extractor;
    }

    protected override Task<BaseFunction> RunCoreAsync(IEGraph input, RunPassContext context)
    {
        var post = (BaseFunction)_extractor.Extract(input.Root!, input);
        IRHelpers.DCE(post);
        return Task.FromResult(post);
    }

    protected override Task OnPassStartAsync(IEGraph input, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            using var fs = DumpScope.Current.OpenFile(Path.Combine("Start", $"V{input.Version}.dot"));
            EGraphPrinter.DumpEgraphAsDot(input, null, fs);
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override Task OnPassEndAsync(BaseFunction post, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            DumpScope.Current.DumpIR(post, "End");
        }

        return Task.CompletedTask;
    }
}
