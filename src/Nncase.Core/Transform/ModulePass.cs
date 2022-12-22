// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform;

/// <summary>
/// Pass in Callable scope.
/// </summary>
public abstract class ModulePass : BasePass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ModulePass"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public ModulePass(string name)
        : base(name)
    {
    }

    /// <summary>
    /// Run current pass for module.
    /// </summary>
    /// <param name="module">Target module.</param>
    /// <param name="options">Options.</param>
    /// <returns><placeholder>A <see cref="Task"/> representing the asynchronous operation.</placeholder></returns>
    public async Task RunAsync(IRModule module, RunPassOptions options)
    {
        var new_options = options.IndentDir(Name);
        OnPassStart(module, new_options);
        await RunCoreAsync(module, new_options);
        OnPassEnd(module, new_options);
    }

    /// <summary>
    /// Run pass implementation for derived class.
    /// </summary>
    /// <param name="module">Target module.</param>
    /// <param name="options">Options.</param>
    /// <returns><placeholder>A <see cref="Task"/> representing the asynchronous operation.</placeholder></returns>
    protected abstract Task RunCoreAsync(IRModule module, RunPassOptions options);

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="module"> module without run pass.</param>
    /// <param name="options"></param>
    protected virtual void OnPassStart(IRModule module, RunPassOptions options)
    {
        if (options.DumpLevel < 3)
        {
            return;
        }

        foreach (var (func, i) in module.Functions.Select((func, i) => (func, i)))
        {
            CompilerServices.DumpIR(func, $"fn_{i}", Path.Combine(options.DumpDir, "Start"));
        }
    }

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="module"> module with rewrited. </param>
    /// <param name="options"></param>
    protected virtual void OnPassEnd(IRModule module, RunPassOptions options)
    {
        if (options.DumpLevel < 3)
        {
            return;
        }

        foreach (var (func, i) in module.Functions.Select((func, i) => (func, i)))
        {
            CompilerServices.DumpIR(func, $"fn_{i}", Path.Combine(options.DumpDir, "End"));
        }
    }
}
