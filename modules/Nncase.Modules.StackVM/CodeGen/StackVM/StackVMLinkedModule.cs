// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

internal class StackVMLinkedModule : ILinkedModule
{
    public StackVMLinkedModule(IReadOnlyList<LinkedFunction> functions, Stream text, Stream? rdata, Stream? custom_calls)
    {
        Functions = functions;
        Sections = new[]
        {
            new LinkedSection(text, ".text", 0, 8, (uint)text.Length),
            new LinkedSection(rdata, ".rdata", 0, 8, (uint)(rdata?.Length ?? 0)),
            new LinkedSection(custom_calls, ".custom_calls", 0, 8, (uint)(custom_calls?.Length ?? 0)),
        };
    }

    public string ModuleKind => StackVMRTModule.Kind;

    public uint Version => StackVMRTModule.Version;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
