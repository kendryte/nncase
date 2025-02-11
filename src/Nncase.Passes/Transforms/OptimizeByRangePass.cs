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
        Add<FoldConstCall>();
        Add<InferRange>();
        Add<FoldNopAbsByRange>();
        Add<FoldNopCompareByRange>();
        Add<FoldNopIf>();
        Add<FoldNopSelect>();
        Add<FoldNopBinary>();
        Add<FoldSameBinary>();
        Add<FoldNopWhere>();
        Add<InlineFunction>(20);
    }
}
