// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.K210;

namespace Nncase.CodeGen.K210;

internal class KPULinkedModule : ILinkedModule
{
    public KPULinkedModule(IReadOnlyList<LinkedFunction> functions, byte[] text, byte[]? rdata)
    {
        Functions = functions;
        Sections = new[]
        {
            new LinkedSection(text, ".text", 0, 8, (ulong)text.Length),
            new LinkedSection(rdata, ".rdata", 0, 8, (ulong)(rdata?.Length ?? 0)),
        };
    }

    public string ModuleKind => KPURTModule.Kind;

    public uint Version => KPURTModule.Version;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
