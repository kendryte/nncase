// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Hosting;
using Nncase.IR;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.PatternMatch.F;
using Tensorflow;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;
using Function = Nncase.IR.Function;
using Math = Nncase.PatternMatch.F.Math;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestMutator
{
    [Fact]
    public void TestMutator()
    {
        Mutator.UnRollLoopSequential();
        Mutator.FoldLet();
        Mutator.UnFoldBlock();
        Mutator.FlattenSequential();
        Mutator.FoldIfThen();
        Mutator.RemoveNop();
        Mutator.FoldMathCall();
    }
}
