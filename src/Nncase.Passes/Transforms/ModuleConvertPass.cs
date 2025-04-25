// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using DryIoc.ImTools;
using NetFabric.Hyperlinq;
using Nncase.Graphs;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes.GraphPartition;
using Nncase.Targets;
using QuikGraph;
using QuikGraph.Algorithms;
using QuikGraph.Graphviz;

namespace Nncase.Passes.Transforms;

public sealed class ModuleConvertPass : ModulePass
{
    public ModuleConvertPass(IModuleCompiler moduleCompiler)
    {
        ModuleCompiler = moduleCompiler;
    }

    public IModuleCompiler ModuleCompiler { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext context)
    {
        var funcs = module.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            if (module.Functions[i] is not Function function)
            {
                continue;
            }

            Function pre = function;
            var postBody = PerformConvert(module, pre.Name, pre);
            var post = pre.With(pre.Name, pre.ModuleKind, postBody, pre.Parameters.ToArray());
            module.Replace(i, post);
        }

        return Task.FromResult(module);
    }

    private Expr PerformConvert(IRModule module, string funcName, Function pre)
    {
        var dynamicVars = IRHelpers.GetDynamicDimVars();
        var parameters = new List<Var>();
        foreach (var item in dynamicVars)
        {
            parameters.Add(item);
        }

        // FIXME: use shapebucket DynamicVar
        foreach (var item in pre.Parameters.ToArray())
        {
            parameters.Add(item);
        }

        var func = new Function($"{funcName}_kernel", ModuleCompiler.ModuleKind, pre.Body, parameters.ToArray());
        module.Add(func);
        return new Call(func, parameters.ToArray());
    }
}
