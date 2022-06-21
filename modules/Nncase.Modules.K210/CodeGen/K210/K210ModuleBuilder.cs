// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.K210;
using Nncase.Runtime.K210;

namespace Nncase.CodeGen.K210;

/// <summary>
/// K210 module builder.
/// </summary>
public class K210ModuleBuilder : ModuleBuilder
{
    /// <inheritdoc/>
    public override string ModuleKind => K210RTModule.Kind;

    /// <inheritdoc/>
    protected override FunctionBuilder CreateFunctionBuilder(uint id)
    {
        return new K210FunctionBuilder(id, SectionManager);
    }

    /// <inheritdoc/>
    protected override ILinkableModule CreateLinkableModule(IReadOnlyList<ILinkableFunction> linkableFunctions)
    {
        return new K210LinkableModule(linkableFunctions, SectionManager);
    }
}
