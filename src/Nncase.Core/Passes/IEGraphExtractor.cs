// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

public interface IEGraphExtractor
{
    public Evaluator.ICostEvaluateProvider CostEvaluateProvider { get; set; }

    Expr Extract(EClass root, IEGraph eGraph);
}
