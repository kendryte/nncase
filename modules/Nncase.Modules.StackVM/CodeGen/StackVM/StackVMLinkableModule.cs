// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.StackVM;

internal class StackVMLinkableModule : LinkableModule
{
    public StackVMLinkableModule(IReadOnlyList<ILinkableFunction> functions, SectionManager sectionManager)
        : base(functions, sectionManager)
    {
    }

    protected override ILinkedModule CreateLinkedModule(IReadOnlyList<LinkedFunction> linkedFunctions, byte[] text)
    {
        return new StackVMLinkedModule(linkedFunctions, text, SectionManager.GetContent(WellknownSectionNames.Rdata));
    }
}
