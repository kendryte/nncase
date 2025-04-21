// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.CostModel;
using Nncase.IR;

namespace Nncase.IR.F;

public partial class CustomCPU
{
    public static Expr MatMul(Expr lhs, Expr rhs, IRArray<int> lhsPackedAxes, IRArray<int> lhsPadedNums, IRArray<int> rhsPackedAxes, IRArray<int> rhsPadedNums, bool transA, bool transB, IRArray<SBP> lhsSBPs, IRArray<SBP> rhsSBPs, IRArray<SBP> outSBPs, Cost cost, string cSourcePath)
    {
        return new Call(new IR.CustomCPU.MatMul(lhsPackedAxes, lhsPadedNums, rhsPackedAxes, rhsPadedNums, transA, transB, lhsSBPs, rhsSBPs, outSBPs, cost, cSourcePath), lhs, rhs);
    }
}
