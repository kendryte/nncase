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
/// the basic pass.
/// </summary>
public abstract class BasePass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="BasePass"/> class.
    /// the base pass ctor.
    /// </summary>
    /// <param name="name"></param>
    public BasePass(string name)
    {
        Name = name;
    }

    /// <summary>
    /// Gets the pass name.
    /// </summary>
    public string Name { get; init; }
}

/// <summary>
/// Pass in Callable scope.
/// </summary>
public abstract class FunctionPass : BasePass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPass"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public FunctionPass(string name)
        : base(name)
    {
    }

    /// <summary>
    /// Run current pass for specific function.
    /// </summary>
    /// <param name="callable">Target function.</param>
    /// <param name="options">Options.</param>
    /// <returns><placeholder>A <see cref="Task"/> representing the asynchronous operation.</placeholder></returns>
    public async Task<BaseFunction> RunAsync(BaseFunction callable, RunPassOptions options)
    {
        var new_options = options.IndentDir(Name).IndentDir(callable.Name);
        OnPassStart(callable, new_options);
        var post = await RunCoreAsync(callable, new_options);
        OnPassEnd(post, new_options);
        return post;
    }

    /// <summary>
    /// Run pass implementation for derived class.
    /// </summary>
    /// <param name="callable">Target function.</param>
    /// <param name="options">Options.</param>
    /// <returns><placeholder>A <see cref="Task"/> representing the asynchronous operation.</placeholder></returns>
    protected abstract Task<BaseFunction> RunCoreAsync(BaseFunction callable, RunPassOptions options);

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="callable"> func without run pass.</param>
    /// <param name="options"></param>
    protected virtual void OnPassStart(BaseFunction callable, RunPassOptions options)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                CompilerServices.DumpIR(callable, "Start", options.DumpDir);
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }

    /// <summary>
    /// the callback function you can custom process func with run pass options.
    /// </summary>
    /// <param name="callable"> func with rewrited. </param>
    /// <param name="options"></param>
    protected virtual void OnPassEnd(BaseFunction callable, RunPassOptions options)
    {
        switch (options.DumpLevel)
        {
            case >= 2:
                CompilerServices.DumpIR(callable, "End", options.DumpDir);
                break;
            case >= 1:
                break;
            default:
                break;
        }
    }
}
