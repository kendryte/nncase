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
public record OpPattern<TOp>(Func<TOp, bool> Condition) : Pattern<TOp>, IOpPattern
    where TOp : Op
{
    /// <summary>
    /// Initializes a new instance of the <see cref="OpPattern{TOp}"/> class.
    /// </summary>
    /// <param name="op">Op expression.</param>
    public OpPattern(TOp op)
        : this(x => x.Equals(op))
    {
    }

    /// <inheritdoc/>
    public Type OpType => typeof(Op);

    /// <inheritdoc/>
    protected override bool MatchLeafCore(TOp expr) => Condition(expr);
}
