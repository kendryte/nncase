// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Passes;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestMutator
{
    [Fact]
    public void TestMutator()
    {
        var unRollLoopSequential = Mutator.UnRollLoopSequential().Invoke();
        Assert.Equal(new Passes.Mutators.UnRollLoopSequential().IsMutated, unRollLoopSequential.IsMutated);

        var foldLet = Mutator.FoldLet().Invoke();
        Assert.Equal(new Passes.Mutators.FoldLet().IsMutated, foldLet.IsMutated);

        var unFoldBlock = Mutator.UnFoldBlock().Invoke();
        Assert.Equal(new Passes.Mutators.UnFoldBlock().IsMutated, unFoldBlock.IsMutated);

        var flattenSequential = Mutator.FlattenSequential().Invoke();
        Assert.Equal(new Passes.Mutators.FlattenSequential().IsMutated, flattenSequential.IsMutated);

        var foldIfThen = Mutator.FoldIfThen().Invoke();
        Assert.Equal(new Passes.Mutators.FoldIfThen().IsMutated, foldIfThen.IsMutated);

        var removeNop = Mutator.RemoveNop().Invoke();
        Assert.Equal(new Passes.Mutators.RemoveNop().IsMutated, removeNop.IsMutated);

        var foldMathCall = Mutator.FoldMathCall().Invoke();
        Assert.Equal(new Passes.Mutators.FoldMathCall().IsMutated, foldMathCall.IsMutated);
    }
}
