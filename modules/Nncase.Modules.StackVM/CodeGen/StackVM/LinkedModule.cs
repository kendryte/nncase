// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

internal class LinkedModule : ILinkedModule
{
    public LinkedModule(IReadOnlyList<LinkedFunction> functions, byte[] text, byte[] rdata)
    {
        Functions = functions;
        Sections = new[]{
            new LinkedSection(text, ".text", 0, 8, (uint)text.Length),
            new LinkedSection(rdata, ".rdata", 0, 8, (uint)rdata.Length)
        };
    }

    public string ModuleKind => StackVMRTModule.Kind;

    public uint Version => StackVMRTModule.Version;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
