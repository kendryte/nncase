// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Runtime.K210;

namespace Nncase.CodeGen.K210;

/// <summary>
/// K210 function builder.
/// </summary>
internal class KPUFunctionBuilder : FunctionBuilder
{
    public KPUFunctionBuilder(uint id, SectionManager sectionManager)
        : base(id, sectionManager)
    {
    }

    protected override void Compile(Callable callable)
    {
    }

    protected override ILinkableFunction CreateLinkableFunction(uint id, Callable callable, IReadOnlyList<FunctionRef> functionRefs, byte[] text)
    {
        return new KPULinkableFunction(id, (Function)callable, functionRefs, text);
    }

    protected override void WriteText()
    {
    }
}
