// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform;

/// <summary>
/// Passes addable.
/// </summary>
public interface IPassesAddable
{
    /// <summary>
    /// Add pass.
    /// </summary>
    /// <typeparam name="T">Pass type.</typeparam>
    /// <param name="parameters">Pass constructor parameters.</param>
    /// <returns>Add result.</returns>
    AddPassResult<T> Add<T>(params object[] parameters)
        where T : class, IPass;

    /// <summary>
    /// Add pass with name.
    /// </summary>
    /// <typeparam name="T">Pass type.</typeparam>
    /// <param name="name">Pass name.</param>
    /// <param name="parameters">Pass constructor parameters.</param>
    /// <returns>Add result.</returns>
    AddPassResult<T> AddWithName<T>(string name, params object[] parameters)
        where T : class, IPass;
}

/// <summary>
/// Pass manager.
/// </summary>
public interface IPassManager : IPassesAddable
{
    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets passes.
    /// </summary>
    public IReadOnlyList<IPass> Passes { get; }

    /// <summary>
    /// Run passes and update the module.
    /// </summary>
    /// <param name="module">Input module.</param>
    /// <returns>A <see cref="Task{IRModule}"/> representing the asynchronous operation.</returns>
    Task<IRModule> RunAsync(IRModule module);
}

/// <summary>
/// Add pass result.
/// </summary>
/// <typeparam name="T">Pass type.</typeparam>
public struct AddPassResult<T> : IPassesAddable
    where T : class, IPass
{
    private readonly PassManager _passManager;

    internal AddPassResult(PassManager passManager, T pass)
    {
        _passManager = passManager;
        Pass = pass;
    }

    /// <summary>
    /// Gets pass.
    /// </summary>
    public T Pass { get; }

    /// <inheritdoc/>
    public AddPassResult<TPass> Add<TPass>(params object[] parameters)
        where TPass : class, IPass => _passManager.Add<TPass>(parameters);

    /// <inheritdoc/>
    public AddPassResult<TPass> AddWithName<TPass>(string name, params object[] parameters)
        where TPass : class, IPass => _passManager.AddWithName<TPass>(name, parameters);

    /// <summary>
    /// Configure pass.
    /// </summary>
    /// <param name="configureRule">Configure pass action.</param>
    /// <returns>This add result.</returns>
    public AddPassResult<T> Configure(Action<T> configureRule)
    {
        configureRule(Pass);
        return this;
    }
}

internal sealed class PassManager : IPassManager
{
    private readonly CompileSession _compileSession;
    private readonly IDumpper _dummper;
    private readonly List<IPass> _passes = new List<IPass>();

    /// <summary>
    /// Initializes a new instance of the <see cref="PassManager"/> class.
    /// </summary>
    /// <param name="name">Pass manager name.</param>
    /// <param name="compileSession">Compile session.</param>
    public PassManager(string name, CompileSession compileSession)
    {
        Name = name;
        _compileSession = compileSession;
        _dummper = DumpScope.GetCurrent(compileSession).CreateSubDummper(name);
    }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets passes.
    /// </summary>
    public IReadOnlyList<IPass> Passes => _passes;

    /// <inheritdoc/>
    public AddPassResult<T> Add<T>(params object[] parameters)
        where T : class, IPass
    {
        using var scope = new CompileSessionScope(_compileSession);
        using var dumpScope = new DumpScope(_dummper);
        var pass = ActivatorUtilities.CreateInstance<T>(_compileSession, parameters);
        _passes.Add(pass);
        return new(this, pass);
    }

    /// <inheritdoc/>
    public AddPassResult<T> AddWithName<T>(string name, params object[] parameters)
        where T : class, IPass
    {
        using var scope = new CompileSessionScope(_compileSession);
        using var dumpScope = new DumpScope(_dummper);
        var result = Add<T>(parameters);
        result.Configure(p => p.Name = name);
        return result;
    }

    /// <inheritdoc/>
    public async Task<IRModule> RunAsync(IRModule module)
    {
        using var dumpScope = new DumpScope(_dummper);
        for (int i = 0; i < _passes.Count; i++)
        {
            var task = _passes[i] switch
            {
                FunctionPass fp => RunAsync(module, i, fp),
                PrimFuncPass pfp => RunAsync(module, i, pfp),
                ModulePass mp => RunAsync(module, i, mp),
                _ => throw new NotSupportedException($"Unsupported pass type: {_passes[i].GetType().AssemblyQualifiedName}"),
            };

            module = await task;
        }

        return module;
    }

    private async Task<IRModule> RunAsync(IRModule module, int passIndex, FunctionPass pass)
    {
        for (int i = 0; i < module.Functions.Count; i++)
        {
            var pre = module.Functions[i];
            var context = new RunPassContext { Index = passIndex };
            var post = await pass.RunAsync(pre, context);
            if (!object.ReferenceEquals(pre, post))
            {
                module.Replace(i, post);
            }
        }

        return module;
    }

    private async Task<IRModule> RunAsync(IRModule module, int passIndex, PrimFuncPass pass)
    {
        for (int i = 0; i < module.Functions.Count; i++)
        {
            var pre = module.Functions[i];
            if (pre is PrimFunction pf)
            {
                var context = new RunPassContext { Index = passIndex };
                var post = await pass.RunAsync(pf, context);
                if (!object.ReferenceEquals(pre, post))
                {
                    module.Replace(i, post);
                }
            }
        }

        return module;
    }

    private async Task<IRModule> RunAsync(IRModule module, int passIndex, ModulePass pass)
    {
        var context = new RunPassContext { Index = passIndex };
        return await pass.RunAsync(module, context);
    }
}
