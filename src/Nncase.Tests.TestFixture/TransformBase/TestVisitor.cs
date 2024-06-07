// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;

namespace Nncase.Tests;

/// <summary>
/// the visitor for test.
/// </summary>
public sealed class TestVisitor : ExprVisitor<bool, IRType>
{
    public TestVisitor(bool visitOtherFunctions = false)
        : base(visitOtherFunctions)
    {
    }

    /// <summary>
    /// check Contains expr with type.
    /// </summary>
    /// <typeparam name="T">the expr type.</typeparam>
    /// <returns>wether contains.</returns>
    public bool Contains<T>()
      where T : Expr
    {
        return ExprMemo.Keys.OfType<T>().Any();
    }

    /// <summary>
    /// count call op numbers.
    /// </summary>
    /// <typeparam name="T">op type.</typeparam>
    /// <returns>counts.</returns>
    public int CountCallOp<T>()
      where T : Op
    {
        var count = ExprMemo.Keys.OfType<Call>().Where(call => call is { Target: T }).Count();
        return count;
    }

    public int CountCallFusion<T>()
      where T : Fusion
    {
        var count = ExprMemo.Keys.OfType<Call>().Where(call => call is { Target: T }).Count();
        return count;
    }

    /// <inheritdoc/>
    protected override bool DefaultVisitLeaf(Expr expr) => true;
}
