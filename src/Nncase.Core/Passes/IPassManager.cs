// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

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
    /// Run passes and update the module.
    /// </summary>
    /// <param name="module">Input module.</param>
    /// <returns>A <see cref="Task{IRModule}"/> representing the asynchronous operation.</returns>
    Task<IRModule> RunAsync(IRModule module);
}

/// <summary>
/// Pass manager factory.
/// </summary>
public interface IPassManagerFactory
{
    /// <summary>
    /// Create pass manager.
    /// </summary>
    /// <param name="name">Pass manager name.</param>
    /// <param name="compileSession">Compile session.</param>
    IPassManager Create(string name, CompileSession compileSession);
}

/// <summary>
/// Add pass result.
/// </summary>
/// <typeparam name="T">Pass type.</typeparam>
public struct AddPassResult<T> : IPassesAddable
    where T : class, IPass
{
    private readonly IPassManager _passManager;
    private readonly CompileSession _compileSession;

    public AddPassResult(IPassManager passManager, CompileSession compileSession, T pass)
    {
        _passManager = passManager;
        _compileSession = compileSession;
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
        using var scope = new CompileSessionScope(_compileSession);
        configureRule(Pass);
        return this;
    }
}
