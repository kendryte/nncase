// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes;

/// <summary>
/// the static mutator can create the mutator in the tir pass.
/// </summary>
public static class Mutator
{
    /// <summary>
    /// Unroll Loop.
    /// </summary>
    public static Func<ExprRewriter> UnRollLoopSequential() => () => new Mutators.UnRollLoopSequential();

    /// <summary>
    /// fold let when expression is const.
    /// </summary>
    public static Func<ExprRewriter> FoldLet() => () => new Mutators.FoldLet();

    /// <summary>
    /// unfold block statements.
    /// </summary>
    public static Func<ExprRewriter> UnFoldBlock() => () => new Mutators.UnFoldBlock();

    /// <summary>
    /// flatten the sequential.
    /// </summary>
    public static Func<ExprRewriter> FlattenSequential() => () => new Mutators.FlattenSequential();

    /// <summary>
    /// substitute.
    /// </summary>
    public static Func<ExprRewriter> Substitute(Func<Expr, Expr?> maper) => () => new Mutators.Substitutor(maper);

    /// <summary>
    /// fold if then else block.
    /// </summary>
    public static Func<ExprRewriter> FoldIfThen() => () => new Mutators.FoldIfThen();

    /// <summary>
    /// 删除内部的T.Nop.
    /// </summary>
    /// <returns>RemoveNop.</returns>
    public static Func<ExprRewriter> RemoveNop() => () => new Mutators.RemoveNop();
}
