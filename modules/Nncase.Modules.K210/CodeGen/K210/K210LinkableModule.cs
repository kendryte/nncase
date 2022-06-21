// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.K210;

internal class K210LinkableModule : LinkableModule
{
    public K210LinkableModule(IReadOnlyList<ILinkableFunction> functions, SectionManager sectionManager)
        : base(functions, sectionManager)
    {
    }

    protected override ILinkedModule CreateLinkedModule(IReadOnlyList<LinkedFunction> linkedFunctions, byte[] text)
    {
        return new K210LinkedModule(linkedFunctions, text, SectionManager.GetContent(WellknownSectionNames.Rdata));
    }
}
