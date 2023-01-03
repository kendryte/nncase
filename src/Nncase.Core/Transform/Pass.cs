// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform;

/// <summary>
/// IR or TIR transformer.
/// </summary>
public interface IPass
{
    /// <summary>
    /// Gets pass name.
    /// </summary>
    string Name { get; }
}

internal interface IPassIntern : IPass
{
    void SetName(string name);
}

/// <summary>
/// IR or TIR transformer.
/// </summary>
/// <typeparam name="T">Type to transform.</typeparam>
public abstract class Pass<T> : IPass, IPassIntern
    where T : class
{
    private string? _name;

    /// <summary>
    /// Initializes a new instance of the <see cref="Pass{T}"/> class.
    /// </summary>
    internal Pass()
    {
        CompileSession = CompileSessionScope.Current;
    }

    /// <inheritdoc/>
    public string Name => _name ??= GetType().Name;

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
        await OnPassStartAsync(input, context);
        var output = await RunCoreAsync(input, context);
        await OnPassEndAsync(output, context);
        return output;
    }

    void IPassIntern.SetName(string name) => _name = name;

    internal virtual string? GetDumpRelativePass(T input) => null;

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
}
