// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;

namespace Nncase.Passes;

internal interface IPassIntern : IPass
{
    CompileSession CompileSession { get; }
}

/// <summary>
/// IR or TIR pass.
/// </summary>
/// <typeparam name="TInput">Type to accept.</typeparam>
/// <typeparam name="TOutput">Type of result.</typeparam>
public abstract class Pass<TInput, TOutput> : IPassIntern
    where TInput : class
    where TOutput : class
{
    private string? _name;

    /// <summary>
    /// Initializes a new instance of the <see cref="Pass{TInput, TOutput}"/> class.
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

    /// <inheritdoc/>
    public virtual IReadOnlyCollection<Type> AnalysisTypes => Array.Empty<Type>();

    CompileSession IPassIntern.CompileSession => CompileSession;

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
    public async Task<TOutput> RunAsync(TInput input, RunPassContext context)
    {
        using var sessionScope = new CompileSessionScope(CompileSession);
        using var dumpScope = new DumpScope(Path.Join($"{context.Index}_{Name}", GetDumpRelativePath(input)));

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
    protected abstract Task<TOutput> RunCoreAsync(TInput input, RunPassContext context);

    /// <summary>
    /// The callback function you can custom process func with run pass context.
    /// </summary>
    /// <param name="input">Input object.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected abstract Task OnPassStartAsync(TInput input, RunPassContext context);

    /// <summary>
    /// The callback function you can custom process func with run pass context.
    /// </summary>
    /// <param name="post">Post object.</param>
    /// <param name="context">Run pass context.</param>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    protected abstract Task OnPassEndAsync(TOutput post, RunPassContext context);

    private protected virtual string? GetDumpRelativePath(TInput input) => null;
}
