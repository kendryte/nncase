// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record TuplePattern(VArgsPattern Fields) : ExprPattern
    {
        public TuplePattern(IR.Tuple tuple) : this(
          new FixedVArgsPattern((from f in tuple.Fields select (ExprPattern)f).ToArray()))
        { }

        public bool MatchLeaf(IR.Tuple tuple)
        {
            return MatchCheckedType(tuple);
        }

    }
    public static partial class Utility
    {
        public static TuplePattern IsTuple(params ExprPattern[] Fields) => new TuplePattern(new FixedVArgsPattern(Fields));

        public static TuplePattern IsTuple(VArgsPattern Fields) => new TuplePattern(Fields);

    }
}