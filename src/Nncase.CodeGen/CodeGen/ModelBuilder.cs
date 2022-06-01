// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

internal class LinkContext : ILinkContext
{
    private readonly IDictionary<Callable, FunctionId> _functionIds;

    public LinkContext(IDictionary<Callable, FunctionId> functionIds)
    {
        _functionIds = functionIds;
    }

    public FunctionId GetFunctionId(Callable function)
    {
        return _functionIds[function];
    }
}

public sealed class ModelBuilder
{
    public ModelBuilder(ITarget target)
    {
        Target = target;
    }

    public ITarget Target { get; }

    public LinkedModel Build(IRModule module)
    {
        var functionsByKind = module.Callables.GroupBy(x => x.ModuleKind).ToList();
        var functionIds = MakeFunctionsIds(functionsByKind);
        var linkableModules = functionsByKind.Select(x => Target.CreateModuleBuilder(x.Key).Build(x.ToList())).ToList();
        var linkContext = new LinkContext(functionIds);
        var linkedModules = linkableModules.Select(x => x.Link(linkContext)).ToList();
        var entryFunctionId = module.Entry == null ? null : functionIds[module.Entry];
        return new LinkedModel(entryFunctionId, linkedModules);
    }

    private Dictionary<Callable, FunctionId> MakeFunctionsIds(IEnumerable<IGrouping<string, Callable>> functionsByKind)
    {
        var ids = new Dictionary<Callable, FunctionId>(ReferenceEqualityComparer.Instance);
        uint moduleId = 0;
        foreach (var fp in functionsByKind)
        {
            uint funcId = 0;
            foreach (var func in fp)
            {
                ids.Add(func, new FunctionId(funcId++, moduleId));
            }

            moduleId++;
        }

        return ids;
    }
}
