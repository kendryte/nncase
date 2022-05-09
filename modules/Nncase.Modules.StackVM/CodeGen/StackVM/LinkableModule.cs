// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

internal class LinkableModule : ILinkableModule
{
    private readonly byte[] _rdata;
    private readonly IReadOnlyList<LinkableFunction> _functions;

    public LinkableModule(byte[] rdata, IReadOnlyList<LinkableFunction> functions)
    {
        _rdata = rdata;
        _functions = functions;
    }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        throw new NotImplementedException();
    }
}
