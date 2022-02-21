// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="Const"/>.
/// </summary>
/// <param name="Fields">Fields condition.</param>
public sealed record TuplePattern(VArgsPattern Fields) : Pattern<IR.Tuple>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TuplePattern"/> class.
    /// </summary>
    /// <param name="tuple"><see cref="IR.Tuple"/> expression.</param>
    public TuplePattern(IR.Tuple tuple)
        : this(new VArgsPattern(tuple.Fields))
    {
    }
}

public static partial class Utility
{
    public static TuplePattern IsTuple(params ExprPattern[] Fields) => new TuplePattern(new VArgsPattern(Fields));

    public static TuplePattern IsTuple(VArgsPattern Fields) => new TuplePattern(Fields);

    public static TuplePattern IsTuple() => IsTuple(IsVArgsRepeat(IsWildcard));

    public static TuplePattern IsConstTuple() => IsTuple(IsVArgsRepeat(() => IsConst()));
}
