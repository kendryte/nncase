// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Evaluator;
using Nncase.Passes;

namespace Nncase.Quantization;

/// <summary>
/// the quantization egraph pass.
/// </summary>
public class EGraphPassWithAdaRound : EGraphRulesPass
{
    /// <inheritdoc/>
    protected override async Task<IEGraph> RunCoreAsync(IEGraph input, RunPassContext context)
    {
        var quantizerAdaRound = new QuantizerAdaRound(input, CompileSession);
        await quantizerAdaRound.RunAsync();
        return input;
    }
}
