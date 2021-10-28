// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record TuplePattern(ID Id, VArgsPattern Fields) : ExprPattern(Id)
    {
        public TuplePattern(IR.Tuple tuple) : this(
          Utility.GetID(),
          new VArgsPattern((from f in tuple.Fields select (ExprPattern)f).ToArray()))
        { }

        public bool MatchLeaf(IR.Tuple tuple)
        {
            return MatchCheckedType(tuple);
        }

    }
    public static partial class Utility
    {
        public static TuplePattern IsTuple(ID Id, params ExprPattern[] Fields) => new TuplePattern(Id, new VArgsPattern(Fields));

        public static TuplePattern IsTuple(params ExprPattern[] Fields) => IsTuple(GetID(), Fields);

        public static TuplePattern IsTuple(ID Id, VArgsPattern Fields) => new TuplePattern(Id, Fields);

        public static TuplePattern IsTuple(VArgsPattern Fields) => IsTuple(GetID(), Fields);
    }
}