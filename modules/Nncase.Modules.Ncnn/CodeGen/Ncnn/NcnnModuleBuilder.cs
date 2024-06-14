// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.Ncnn;

/// <summary>
/// Ncnn module builder.
/// </summary>
public class NcnnModuleBuilder : ModuleBuilder
{
    /// <inheritdoc/>
    public override string ModuleKind => "ncnn";

    /// <inheritdoc/>
    protected override ILinkableModule CreateLinkableModule(IReadOnlyList<ILinkableFunction> linkableFunctions)
    {
        return new NcnnLinkableModule(linkableFunctions, SectionManager);
    }

    /// <inheritdoc/>
    protected override FunctionBuilder CreateFunctionBuilder(uint id)
    {
        return new NcnnFunctionBuilder(id, SectionManager);
    }
}
