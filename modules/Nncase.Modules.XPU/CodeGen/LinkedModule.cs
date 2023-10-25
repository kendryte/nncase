// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.XPU;

internal sealed class LinkedModule : ILinkedModule
{
    public LinkedModule(IReadOnlyList<ILinkedFunction> functions, Stream text, Stream rdata)
    {
        Functions = functions;
        Sections = new[]
        {
            new LinkedSection(text, WellknownSectionNames.Text, 0, 8, (ulong)text.Length),
            new LinkedSection(rdata, WellknownSectionNames.Rdata, 0, 8, (ulong)rdata.Length),
        };
    }

    public string ModuleKind => Targets.XPUTarget.Kind;

    public uint Version => 0;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
