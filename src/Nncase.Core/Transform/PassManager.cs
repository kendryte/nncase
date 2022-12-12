// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Transform;

/// <summary>
/// Pass manager.
/// </summary>
public class PassManager : IEnumerable<BasePass>
{
    private readonly IRModule _module;
    private readonly RunPassOptions _options;
    private readonly List<BasePass> _passes = new List<BasePass>();

    private Dictionary<BaseFunction, BaseFunction> _functions_update_map = new(ReferenceEqualityComparer.Instance);
    private Dictionary<int, BaseFunction> _functions_mask = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="PassManager"/> class.
    /// </summary>
    /// <param name="module">Module.</param>
    /// <param name="options">Options.</param>
    public PassManager(IRModule module, RunPassOptions options)
    {
        _module = module;
        _options = options;
    }

    /// <summary>
    /// Add function pass.
    /// </summary>
    /// <param name="pass">Pass.</param>
    public void Add(BasePass pass)
    {
        _passes.Add(pass);
    }

    /// <inheritdoc/>
    public IEnumerator<BasePass> GetEnumerator()
    {
        return ((IEnumerable<BasePass>)_passes).GetEnumerator();
    }

    /// <summary>
    /// Run passes and update the module.
    /// </summary>
    public async Task RunAsync()
    {
        var passes = _passes.AsEnumerable();
        while (passes.Count() > 0)
        {
            var type = passes.First().GetType();
            Type base_type;
            if (type.IsSubclassOf(typeof(FunctionPass)))
                base_type = typeof(FunctionPass);
            else if (type.IsSubclassOf(typeof(ModulePass)))
                base_type = typeof(ModulePass);
            else
                throw new ArgumentOutOfRangeException();

            var candiate = passes.TakeWhile(item => item.GetType().IsSubclassOf(base_type));
            passes = passes.Skip(candiate.Count());

            if (type.IsSubclassOf(typeof(FunctionPass)))
                await runFunctionAsync(candiate.OfType<FunctionPass>());
            else if (type.IsSubclassOf(typeof(ModulePass)))
                await runModuleAsync(candiate.OfType<ModulePass>());
            else
                throw new ArgumentOutOfRangeException();
        }
    }

    private async Task runFunctionAsync(IEnumerable<FunctionPass> passes)
    {
        int i = 0;
        string name = string.Empty;
        while (i < _module.Functions.Count)
        {
            foreach (var pass in passes)
            {
                var pre = _module.Functions[i];
                var post = await pass.RunAsync(pre, _options);
                if (!object.ReferenceEquals(pre, post))
                {
                    FuncUpdateRecord(i, pre, post);
                    _module.Update(i, post);
                }

                name = pass.Name;
            }

            i++;
        }

        FuncUpdateDependence(_module, _functions_update_map, _options, name);
        CleanFuncUpdateRecord();
    }

    private async Task runModuleAsync(IEnumerable<ModulePass> passes)
    {
        foreach (var pass in passes)
        {
            await pass.RunAsync(_module, _options);
        }
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable)_passes).GetEnumerator();
    }

    private void CleanFuncUpdateRecord()
    {
        _functions_update_map.Clear();
        _functions_mask.Clear();
    }

    private void FuncUpdateRecord(int i, BaseFunction current, BaseFunction updated)
    {
        // if function[i] has not been update, record it to origin function.
        if (!_functions_mask.TryGetValue(i, out var origin))
        {
            origin = current;
            _functions_mask.Add(i, origin);
        }

        _functions_update_map[origin] = updated;
    }

    /// <summary>
    /// foreach fix when the call target function has been updated.
    /// </summary>
    public static void FuncUpdateDependence(IRModule module, Dictionary<BaseFunction, BaseFunction> update_map, RunPassOptions options, string name)
    {
        var mutator = new DependenceMutator(update_map);
        var post = mutator.Visit(module.Entry!);
        if (!mutator.IsMutated)
            return;

        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (update_map.TryGetValue(module.Functions[i], out var updated_func))
                module.Update(i, updated_func);
        }

        if (options.DumpLevel > 2)
        {
            foreach (var item in module.Functions)
            {
                CompilerServices.DumpIR(item, "", Path.Combine(options.DumpDir, name, $"FuncUpdateDependence"));
            }
        }
    }
}

/// <summary>
/// Update the function call dependencer
/// </summary>
internal sealed class DependenceMutator : DeepExprMutator
{
    public Dictionary<BaseFunction, BaseFunction> functionsUpdated;

    public DependenceMutator(Dictionary<BaseFunction, BaseFunction> functions_updated)
    {
        functionsUpdated = functions_updated;
    }

    public override Expr DefaultMutateLeaf(Expr expr)
    {
        if (expr is BaseFunction baseFunction && functionsUpdated.TryGetValue(baseFunction, out var updated_basefunc))
            return updated_basefunc;
        return expr;
    }

    public override Expr Visit(BaseFunction baseFunction)
    {
        // first time enter function, mutate
        var nexpr = base.Visit(baseFunction);
        if (nexpr is BaseFunction updatedBasefunction && !object.ReferenceEquals(baseFunction, updatedBasefunction))
        {
            if (functionsUpdated.ContainsKey(baseFunction))
                functionsUpdated[baseFunction] = updatedBasefunction;
            else
                functionsUpdated.Add(baseFunction, updatedBasefunction);
        }

        return nexpr;
    }
}