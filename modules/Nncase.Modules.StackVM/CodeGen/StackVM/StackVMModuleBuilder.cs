// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

/// <summary>
/// StackVM module builder.
/// </summary>
public class StackVMModuleBuilder : ModuleBuilder
{
    /// <inheritdoc/>
    public override string ModuleKind => StackVMRTModule.Kind;

    /// <inheritdoc/>
    protected override ILinkableModule CreateLinkableModule(IReadOnlyList<ILinkableFunction> linkableFunctions)
    {
        return new StackVMLinkableModule(linkableFunctions, SectionManager);
    }

    /// <inheritdoc/>
    protected override FunctionBuilder CreateFunctionBuilder(uint id)
    {
        return new StackVMFunctionBuilder(id, SectionManager);
    }
}
