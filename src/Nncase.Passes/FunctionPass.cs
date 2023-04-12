// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.Passes;

/// <summary>
/// Pass in Callable scope.
/// </summary>
public abstract class FunctionPass : Pass<BaseFunction, BaseFunction>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPass"/> class.
    /// </summary>
    public FunctionPass()
    {
    }

    /// <inheritdoc/>
    protected override Task OnPassStartAsync(BaseFunction input, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            DumpScope.Current.DumpIR(input, "Start");
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

    private protected override string? GetDumpRelativePath(BaseFunction input) => input.Name;
}
