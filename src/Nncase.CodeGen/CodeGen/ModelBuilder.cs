// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

/// <summary>
/// The Kmodel Builder.
/// </summary>
public sealed class ModelBuilder : IModelBuilder
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ModelBuilder"/> class.
    /// default ctor.
    /// </summary>
    public ModelBuilder(ITarget target, CompileOptions compileOptions)
    {
        Target = target;
        CompileOptions = compileOptions;
    }

    /// <summary>
    /// Gets get the Target.
    /// </summary>
    public ITarget Target { get; }

    /// <summary>
    /// Gets get the CompileOptions.
    /// </summary>
    public CompileOptions CompileOptions { get; }

    public ILinkedModel Build(IRModule module)
    {
        var functionsByKind = module.Functions.GroupBy(x => x.ModuleKind).ToList();
        var functionIds = MakeFunctionsIds(functionsByKind);
        var linkableModules = functionsByKind.Select(x => Target.CreateModuleBuilder(x.Key, CompileOptions).Build(x.ToList())).ToList();
        var linkContext = new LinkContext(functionIds);
        var linkedModules = linkableModules.Select(x => x.Link(linkContext)).ToList();
        var entryFunctionId = module.Entry == null ? null : functionIds[module.Entry];
        return new LinkedModel(entryFunctionId, linkedModules);
    }

    private Dictionary<BaseFunction, FunctionId> MakeFunctionsIds(IEnumerable<IGrouping<string, BaseFunction>> functionsByKind)
    {
        var ids = new Dictionary<BaseFunction, FunctionId>(ReferenceEqualityComparer.Instance);
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

internal class LinkContext : ILinkContext
{
    private readonly IDictionary<BaseFunction, FunctionId> _functionIds;

    public LinkContext(IDictionary<BaseFunction, FunctionId> functionIds)
    {
        _functionIds = functionIds;
    }

    public FunctionId GetFunctionId(BaseFunction function)
    {
        return _functionIds[function];
    }
}
