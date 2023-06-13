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
/// Module builder.
/// </summary>
public abstract class ModuleBuilder : IModuleBuilder
{
    /// <inheritdoc/>
    public abstract string ModuleKind { get; }

    public SectionManager SectionManager { get; } = new SectionManager();

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var linkableFunctions = Compile(functions);
        return CreateLinkableModule(linkableFunctions);
    }

    protected abstract ILinkableModule CreateLinkableModule(IReadOnlyList<ILinkableFunction> linkableFunctions);

    protected abstract FunctionBuilder CreateFunctionBuilder(uint id);

    private ILinkableFunction[] Compile(IEnumerable<BaseFunction> functions)
    {
        return functions.Select((f, i) => CreateFunctionBuilder((uint)i).Build(f)).ToArray();
    }
}
