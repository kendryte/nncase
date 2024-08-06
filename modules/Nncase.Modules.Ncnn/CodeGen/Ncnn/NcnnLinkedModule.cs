// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.Ncnn;

internal class NcnnLinkedModule : ILinkedModule
{
    public NcnnLinkedModule(IReadOnlyList<LinkedFunction> functions, Stream text, Stream? rdata)
    {
        Functions = functions;
        Sections = new[]
        {
            new LinkedSection(text, ".text", 0, 8, (uint)text.Length),
            new LinkedSection(rdata, ".rdata", 0, 8, (uint)(rdata?.Length ?? 0)),
        };
    }

    public string ModuleKind => "ncnn";

    public uint Version => 1;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
