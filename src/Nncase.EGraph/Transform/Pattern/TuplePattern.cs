// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record TuplePattern(IRArray<ExprPattern> FieldPats) : ExprPattern
    {
        public bool MatchLeaf(IR.Tuple tuple)
        {
            return (FieldPats.Count == tuple.Fields.Count) && MatchCheckedType(tuple);
        }

    }
}