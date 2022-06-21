// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.K210;

namespace Nncase.CodeGen.K210;

internal class K210LinkedModule : ILinkedModule
{
    public K210LinkedModule(IReadOnlyList<LinkedFunction> functions, byte[] text, byte[]? rdata)
    {
        Functions = functions;
        Sections = new[]
        {
            new LinkedSection(text, ".text", 0, 8, (uint)text.Length),
            new LinkedSection(rdata, ".rdata", 0, 8, (uint)(rdata?.Length ?? 0)),
        };
    }

    public string ModuleKind => K210RTModule.Kind;

    public uint Version => K210RTModule.Version;

    public IReadOnlyList<ILinkedFunction> Functions { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
