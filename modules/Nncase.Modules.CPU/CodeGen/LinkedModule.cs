// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.CPU;

internal sealed class LinkedModule : ILinkedModule
{
    public LinkedModule(IReadOnlyList<ILinkedFunction> functions, byte[] text, byte[] rdata)
    {
        Functions = functions;
        Sections = new[] { new LinkedSection(text, ".text", 0, 8, (uint)text.Length) };
    }

    public string ModuleKind => Targets.CPUTarget.Kind;

    public uint Version => 0;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
