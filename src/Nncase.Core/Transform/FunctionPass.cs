// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using static TorchSharp.torch.nn;

namespace Nncase.Transform;

/// <summary>
/// Pass in Callable scope.
/// </summary>
public abstract class FunctionPass : Pass<BaseFunction>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPass"/> class.
    /// </summary>
    public FunctionPass()
    {
    }

    internal override string? GetDumpRelativePass(BaseFunction input) => input.Name;

    /// <inheritdoc/>
    protected override Task OnPassStartAsync(BaseFunction input, RunPassContext context)
    {
        if (context.Dumpper.IsEnabled(DumpFlags.PassIR))
        {
            context.Dumpper.DumpIR(input, "Start");
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override Task OnPassEndAsync(BaseFunction post, RunPassContext context)
    {
        if (context.Dumpper.IsEnabled(DumpFlags.PassIR))
        {
            context.Dumpper.DumpIR(post, "End");
        }

        return Task.CompletedTask;
    }
}
