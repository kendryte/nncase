// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Transform.Pattern.Math;

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
            (BinaryPattern binaryPat, Binary binary) => binaryPat.MatchLeaf(binary),
            (ClampPattern clampPat, Clamp clamp) => clampPat.MatchLeaf(clamp),
            (UnaryPattern unaryPat, Unary unary) => unaryPat.MatchLeaf(unary),
            (_, _) => false
        };
    }
}