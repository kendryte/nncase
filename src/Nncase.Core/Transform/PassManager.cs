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
/// Pass manager.
/// </summary>
public sealed class PassManager
{
    private readonly CompileSession _compileSession;
    private readonly IDumpperFactory _dumpperFactory;
    private readonly List<IPass> _passes = new List<IPass>();

    internal PassManager(string name, CompileSession compileSession, IDumpperFactory dumpperFactory)
    {
        Name = name;
        _compileSession = compileSession;
        _dumpperFactory = dumpperFactory;
    }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets passes.
    /// </summary>
    public IReadOnlyList<IPass> Passes => _passes;

    /// <summary>
    /// Add pass.
    /// </summary>
    /// <typeparam name="T">Pass type.</typeparam>
    /// <param name="name">Pass name.</param>
    /// <param name="configurePass">Configure pass action.</param>
    /// <param name="parameters">Pass constructor parameters.</param>
    /// <returns>This pass manager.</returns>
    public PassManager Add<T>(string name, Action<T>? configurePass = null, params object[] parameters)
        where T : class, IPass
    {
        using var scope = new CompileSessionScope(_compileSession);
        var pass = ActivatorUtilities.CreateInstance<T>(_compileSession.ServiceProvider, parameters);
        ((IPassIntern)pass).SetName(name);
        configurePass?.Invoke(pass);
        _passes.Add(pass);
        return this;
    }

    /// <summary>
    /// Run passes and update the module.
    /// </summary>
    /// <param name="module">Input module.</param>
    /// <returns>A <see cref="Task{IRModule}"/> representing the asynchronous operation.</returns>
    public async Task<IRModule> RunAsync(IRModule module)
    {
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
            var dumpper = _dumpperFactory.CreateDummper(Path.Join(Name, $"{passIndex}_{pass.Name}", pass.GetDumpRelativePass(pre)));
            var context = new RunPassContext(dumpper);
            var post = await pass.RunAsync(pre, context);
            if (!object.ReferenceEquals(pre, post))
            {
                module.Update(i, post);
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
                var dumpper = _dumpperFactory.CreateDummper(Path.Join(Name, $"{passIndex}_{pass.Name}", pass.GetDumpRelativePass(pf)));
                var context = new RunPassContext(dumpper);
                var post = await pass.RunAsync(pf, context);
                if (!object.ReferenceEquals(pre, post))
                {
                    module.Update(i, post);
                }
            }
        }

        return module;
    }

    private async Task<IRModule> RunAsync(IRModule module, int passIndex, ModulePass pass)
    {
        var dumpper = _dumpperFactory.CreateDummper(Path.Join(Name, $"{passIndex}_{pass.Name}", pass.GetDumpRelativePass(module)));
        var context = new RunPassContext(dumpper);
        return await pass.RunAsync(module, context);
    }
}
