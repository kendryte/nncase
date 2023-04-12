// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.K210;

internal class KPULinkableFunction : ILinkableFunction
{
    public KPULinkableFunction(uint id, Function sourceFunction, IEnumerable<FunctionRef> functionRefs, byte[] text)
    {
        Id = id;
        SourceFunction = sourceFunction;
        Text = text;
        FunctionRefs = functionRefs;
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public byte[] Text { get; }

    public IEnumerable<FunctionRef> FunctionRefs { get; }

    public IReadOnlyList<ILinkedSection> Sections => Array.Empty<ILinkedSection>();
}
