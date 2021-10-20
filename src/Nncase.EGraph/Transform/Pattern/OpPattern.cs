// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;

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

    public abstract record OpPattern(IRArray<ParameterInfoPattern> Parameters) : ExprPattern
    {
        public bool MatchLeaf(Op op) => (this, op) switch
        {
            (Math.BinaryPattern binaryPat, Binary binary) => binaryPat.MatchLeaf(binary),
            (Math.ClampPattern clampPat, Clamp clamp) => clampPat.MatchLeaf(clamp),
            (Math.UnaryPattern unaryPat, Unary unary) => unaryPat.MatchLeaf(unary),
            (_, _) => throw new NotImplementedException($"Can't Match Pattern {this.GetType()} and Op {op.GetType()}")
        };
    }
}