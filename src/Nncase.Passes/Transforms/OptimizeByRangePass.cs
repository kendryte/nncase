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
        pass.Add<FoldNopAbsByRange>();
        pass.Add<FoldNopCompareByRange>();
        pass.Add<FoldNopIf>();
        pass.Add<FoldNopSelect>();
        pass.Add<FoldNopBinary>();
        pass.Add<FoldSameBinary>();
        pass.Add<FoldNopWhere>();
        pass.Add<InlineFunction>(20);
    }
}
