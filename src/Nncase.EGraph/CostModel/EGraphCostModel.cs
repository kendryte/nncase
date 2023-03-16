﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Passes;

namespace Nncase.CostModel;

internal class EGraphCostModel
{
    private readonly IReadOnlyDictionary<ENode, Cost> _costs;

    public EGraphCostModel(IReadOnlyDictionary<ENode, Cost> costs)
    {
        _costs = costs;
    }

    public Cost this[ENode enode] => _costs[enode];
}
