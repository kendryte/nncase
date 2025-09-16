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

public partial class CustomNTT
{
    public static Expr MatMul(Expr lhs, Expr rhs, Expr scale, Expr extraWorkload, IRArray<int> lhsPackedAxes, IRArray<int> rhsPackedAxes, bool transA, bool transB, IRArray<SBP> lhsSBPs, IRArray<SBP> rhsSBPs, IRArray<SBP> outSBPs, Cost cost, string cSourcePath, string funcName, DataType outputDataType)
    {
        return new Call(new IR.CustomNTT.MatMul(lhsPackedAxes, rhsPackedAxes, transA, transB, lhsSBPs, rhsSBPs, outSBPs, cost, cSourcePath, funcName, outputDataType), lhs, rhs, scale, extraWorkload);
    }

    public static Expr LayerNorm(Expr input, Expr scale, Expr bias, Expr postScale, int axis, float epsilon, bool useMean, bool channelFirst, int[] vectorizedAxes, IRArray<SBP> inSBPs, IRArray<SBP> scaleSBPs, IRArray<SBP> biasSBPs, IRArray<SBP> outSBPs, Cost cost, string cSourcePath, string funcName, DataType outputDataType)
    {
        return new Call(new IR.CustomNTT.LayerNorm(axis, epsilon, useMean, channelFirst, vectorizedAxes, inSBPs, scaleSBPs, biasSBPs, outSBPs, cost, cSourcePath, funcName, outputDataType), input, scale, bias, postScale);
    }
}
