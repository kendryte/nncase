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
public abstract class ModulePass : Pass<IRModule, IRModule>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ModulePass"/> class.
    /// </summary>
    public ModulePass()
    {
    }

    /// <inheritdoc/>
    protected override Task OnPassStartAsync(IRModule input, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            foreach (var func in input.Functions)
            {
                DumpScope.Current.DumpIR(func, string.Empty, "Start");
            }
        }

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override Task OnPassEndAsync(IRModule post, RunPassContext context)
    {
        if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            foreach (var func in post.Functions)
            {
                DumpScope.Current.DumpIR(func, string.Empty, "End");
            }
        }

        return Task.CompletedTask;
    }
}
