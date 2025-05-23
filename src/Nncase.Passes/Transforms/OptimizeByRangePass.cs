// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeExpr;

namespace Nncase.Passes.Transforms;

public sealed class OptimizeByRangePass : DataflowPass
{
    public OptimizeByRangePass()
    {
        RegisterRules(this);
    }

    public static void RegisterRules(DataflowPass pass)
    {
        pass.Add<FoldConstCall>();
        pass.Add<InferRange>();
        pass.Add<FoldNopCast>();
        pass.Add<FoldClampByRangeConst>();
        pass.Add<FoldClampByRangeVar>();
        pass.Add<FoldNopAbsByRange>();
        pass.Add<FoldNopCompareByRange>();
        pass.Add<FoldNopIf>();
        pass.Add<FoldNopSelect>();
        pass.Add<FoldGetItemShapeOf>();
        pass.Add<SimplifySelect>();
        pass.Add<FoldCompareSelect>();
        pass.Add<SwapBinaryArgs>();
        pass.Add<FoldNopBinary>();
        pass.Add<FoldNopBinaryByRange>();
        pass.Add<FoldSameBinary>();
        pass.Add<SimplifyBinaryCeilDiv>();
        pass.Add<FoldNopWhere>();
        pass.Add<InlineFunction>(20);
    }
}
