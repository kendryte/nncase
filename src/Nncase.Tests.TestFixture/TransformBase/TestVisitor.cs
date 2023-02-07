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
    /// <inheritdoc/>
    public override bool DefaultVisitLeaf(Expr expr) => true;

    /// <summary>
    /// check Contains expr with type.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    public bool Contains<T>()
      where T : Expr
    {
        return ExpressionMemo.Keys.OfType<T>().Any();
    }

    /// <summary>
    /// count call op numbers.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <returns></returns>
    public int CountCallOp<T>()
      where T : Op
    {
        return ExpressionMemo.Keys.OfType<Call>().Where(call => call is { Target: T }).Count();
    }
}
