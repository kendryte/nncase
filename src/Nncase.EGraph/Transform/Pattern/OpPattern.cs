// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record ParameterInfoPattern(Func<ParameterInfo, bool> Cond)
    {
        public bool Match(ParameterInfo x) => Cond(x);

        public ParameterInfoPattern(ParameterInfo Info) : this(info => info == Info)
        {
        }

        public ParameterInfoPattern(string Name) : this(info => info.Name == Name)
        {
        }
    };

    public abstract record OpPattern(IRArray<ParameterInfoPattern> ParameterPats) : ExprPattern
    {
        public OpPattern(Op op) : this(ImmutableArray.Create((from p in op.Parameters select new ParameterInfoPattern(p)).ToArray())) { }

        public bool MatchLeaf(Op op)
        {
            if (ParameterPats.Count != op.Parameters.Count)
            {
                return false;
            }
            foreach (var (ppat, p) in ParameterPats.Zip(op.Parameters))
            {
                if (!ppat.Match(p))
                {
                    return false;
                }
            }
            return true && MatchCheckedType(op);
        }
    }
}