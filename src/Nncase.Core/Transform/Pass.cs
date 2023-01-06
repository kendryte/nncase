// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform;

/// <summary>
/// IR or TIR transformer.
/// </summary>
public interface IPass
{
    /// <summary>
    /// Gets or sets pass name.
    /// </summary>
    string Name { get; set; }
}

/// <summary>
/// IR or TIR transformer.
/// </summary>
/// <typeparam name="T">Type to transform.</typeparam>
public abstract class Pass<T> : IPass
    where T : class
{
    private string? _name;

    /// <summary>
    /// Initializes a new instance of the <see cref="Pass{T}"/> class.
    /// </summary>
    internal Pass()
    {
        CompileSession = CompileSessionScope.GetCurrentThrowIfNull();
    }

    /// <inheritdoc/>
    public string Name
    {
        get => _name ??= GetType().Name;
        set => _name = value;
    }

    /// <summary>
    /// Gets compile session.
    /// </summary>
    protected CompileSession CompileSession { get; }

    /// <summary>
    /// Run pass.
    /// </summary>
    /// <param name="input">Input object.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>Output object.</returns>
    public async Task<T> RunAsync(T input, RunPassContext context)
    {
        using var sessionScope = new CompileSessionScope(CompileSession);
        using var dumpScope = new DumpScope(Path.Join($"{context.Index}_{Name}", GetDumpRelativePass(input)));

        await OnPassStartAsync(input, context);
        var output = await RunCoreAsync(input, context);
        await OnPassEndAsync(output, context);
        return output;
    }

    /// <summary>
    /// Run pass implementation for derived class.
    /// </summary>
    /// <param name="input">Input object.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected abstract Task<T> RunCoreAsync(T input, RunPassContext context);

    /// <summary>
    /// The callback function you can custom process func with run pass context.
    /// </summary>
    /// <param name="input">Input object.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected abstract Task OnPassStartAsync(T input, RunPassContext context);

    /// <summary>
    /// The callback function you can custom process func with run pass context.
    /// </summary>
    /// <param name="post">Post object.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected abstract Task OnPassEndAsync(T post, RunPassContext context);

    private protected virtual string? GetDumpRelativePass(T input) => null;
}
