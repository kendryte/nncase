// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Op pattern interface.
/// </summary>
public interface IOpPattern : IPattern
{
    /// <summary>
    /// Gets op type.
    /// </summary>
    Type OpType { get; }
}

/// <summary>
/// Pattern for <see cref="Op"/>.
/// </summary>
/// <typeparam name="TOp">Op type.</typeparam>
/// <param name="Condition">Condition.</param>
/// <param name="Name">name.</param>
public record OpPattern<TOp>(Func<TOp, bool> Condition, string? Name) : Pattern<TOp>(Name), IOpPattern
    where TOp : Op
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OpPattern{TOp}"/> class.
    /// </summary>
    /// <param name="op">Op expression.</param>
    /// <param name="name">name.</param>
    public OpPattern(TOp op, string? name)
        : this(x => x.Equals(op), name)
    {
    }

    /// <inheritdoc/>
    public Type OpType => typeof(Op);

    /// <inheritdoc/>
    protected override bool MatchLeafCore(TOp expr) => Condition(expr);
}


public static partial class Utility
{
    public static OpPattern<TOp> IsOp<TOp>(string? name, Func<TOp, bool> Condition)
       where TOp : Op
       => new OpPattern<TOp>(Condition, name);

    public static OpPattern<TOp> IsOp<TOp>(Func<TOp, bool> Condition)
       where TOp : Op
        => IsOp<TOp>(Condition);
}