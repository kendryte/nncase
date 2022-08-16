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

namespace Nncase.Transform;

/// <summary>
/// Pass manager.
/// </summary>
public class PassManager : IEnumerable<BasePass>
{
    private readonly IRModule _module;
    private readonly RunPassOptions _options;
    private readonly List<BasePass> _passes = new List<BasePass>();

    private Dictionary<BaseFunction, BaseFunction> _functions_update = new(ReferenceEqualityComparer.Instance);
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
                await runFunctionAsync(candiate);
            else if (type.IsSubclassOf(typeof(ModulePass)))
                await runModuleAsync(candiate);
            else
                throw new ArgumentOutOfRangeException();
        }
    }

    private async Task runFunctionAsync(IEnumerable<BasePass> passes)
    {
        int i = 0;
        while (i < _module.Functions.Count)
        {
            foreach (var pass in passes)
            {
                var post = await ((FunctionPass)pass).RunAsync(_module.Functions[i], _options);
                FuncUpdateRecord(i, _module.Functions[i], post);
                _module.Update(i, post);
            }
            i++;
        }
        FuncUpdateDependence();
        CleanFuncUpdateRecord();
    }

    private async Task runModuleAsync(IEnumerable<BasePass> passes)
    {
        foreach (var pass in passes)
        {
            await ((ModulePass)pass).RunAsync(_module, _options);
        }
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        return ((IEnumerable)_passes).GetEnumerator();
    }


    private void CleanFuncUpdateRecord()
    {
        _functions_update.Clear();
        _functions_mask.Clear();
    }

    private void FuncUpdateRecord(int i, BaseFunction current, BaseFunction function)
    {
        if (!_functions_mask.TryGetValue(i, out var origin))
        {
            origin = current;
            _functions_mask[i] = origin;
        }
        _functions_update[origin] = function;
    }

    /// <summary>
    /// foreach fix when the call target function has been updated.
    /// </summary>
    private void FuncUpdateDependence()
    {
        var mutator = new DependenceMutator(_functions_update);
        var post = mutator.Visit(_module.Entry!);
        if (!mutator.IsMutated)
            return;

        for (int i = 0; i < _module.Functions.Count; i++)
        {
            if (_functions_update.TryGetValue(_module.Functions[i], out var updated_func))
                _module.Update(i, updated_func);
        }
        foreach (var item in _module.Functions)
        {
            CompilerServices.DumpIR(item, "", Path.Combine(_options.DumpDir, "FuncUpdateDependence"));
        }
    }
}

/// <summary>
/// Update the function call dependence
/// NOTE skip visit prim func/fusion/prim function wrapper
/// </summary>
internal sealed class DependenceMutator : ExprMutator
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

    public override Expr Visit(Expr expr)
    {
        var nexpr = base.Visit(expr);
        if (expr is BaseFunction baseFunction && nexpr is BaseFunction updatedBasefunction && !nexpr.Equals(expr))
        {
            if (functionsUpdated.ContainsKey(baseFunction))
                functionsUpdated[baseFunction] = updatedBasefunction;
            else
                functionsUpdated.Add(baseFunction, updatedBasefunction);
        }
        return nexpr;
    }
}