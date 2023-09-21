// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.F;
using Nncase.IR.NN;

namespace Nncase.Importer.Ncnn;

public partial class NcnnImporter
{
    private Expr VisitSplit(NcnnLayer layer)
    {
        var input = GetInputExprs(layer, 0);
        var output = new IR.Tuple(Enumerable.Repeat(input, layer.Tops.Length).ToArray());
        return output;
    }
}
