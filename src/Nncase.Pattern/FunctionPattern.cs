// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Pattern
{
    public sealed record FunctionPattern(ExprPattern Body, VArgsPattern Parameters) : ExprPattern
    {
        public FunctionPattern(Function func) : this(
            (ExprPattern)func.Body,
            new FixedVArgsPattern((from p in func.Parameters select ((ExprPattern)p)).ToArray()))
        { }

        public FunctionPattern(ExprPattern body, params ExprPattern[] parameters) : this(body, new FixedVArgsPattern(parameters))
        {
        }

        public FunctionPattern(ExprPattern body, IRArray<ExprPattern> parameters) : this(body, new FixedVArgsPattern(parameters))
        {
        }

        public bool MatchLeaf(Function func) => MatchCheckedType(func);
    }
    public static partial class Utility
    {
        public static FunctionPattern IsFunction(ExprPattern Body, VArgsPattern Parameters) => new FunctionPattern(Body, Parameters);
    }
}