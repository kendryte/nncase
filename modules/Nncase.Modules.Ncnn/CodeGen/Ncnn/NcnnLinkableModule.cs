// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.Ncnn;

internal class NcnnLinkableModule : LinkableModule
{
    public NcnnLinkableModule(IReadOnlyList<ILinkableFunction> functions, SectionManager sectionManager)
        : base(functions, sectionManager)
    {
    }

    protected override ILinkedModule CreateLinkedModule(IReadOnlyList<LinkedFunction> linkedFunctions, Stream text)
    {
        return new NcnnLinkedModule(
            linkedFunctions,
            text,
            SectionManager.GetContent(WellknownSectionNames.Rdata));
    }
}
