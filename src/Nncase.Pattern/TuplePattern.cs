// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Pattern
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

        public override ExprPattern Copy() =>
          this with { Id = _globalPatIndex++, Fields = Fields.Copy() };

        public override void Clear()
        {
            Fields.Clear();
        }


    }
    public static partial class Utility
    {
        public static TuplePattern IsTuple(params ExprPattern[] Fields) => new TuplePattern(new FixedVArgsPattern(Fields));

        public static TuplePattern IsTuple(VArgsPattern Fields) => new TuplePattern(Fields);

        public static TuplePattern IsTuple() => IsTuple(IsVArgsRepeat((n, param) =>
            {
                for (int i = 0; i < n; i++)
                {
                    param.Add(IsWildCard());
                }
            },
            (match, param) =>
            {
                if (!match)
                {
                    param.Clear();
                }
            }
            ));
        public static TuplePattern IsConstTuple() => IsTuple(IsVArgsRepeat(() => IsConst()));
    }
}