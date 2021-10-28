// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record FunctionPattern(ID Id, ExprPattern Body, VArgsPattern Parameters) : ExprPattern(Id)
    {
        public FunctionPattern(Function func) : this(
            Utility.GetID(),
            (ExprPattern)func.Body,
            new VArgsPattern((from p in func.Parameters select ((ExprPattern)p)).ToArray()))
        { }

        public FunctionPattern(ExprPattern body, VArgsPattern parameters) : this(Utility.GetID(), body, parameters) { }


        public FunctionPattern(ExprPattern body, params ExprPattern[] parameters) : this(Utility.GetID(), body, new VArgsPattern(parameters, null))
        {
        }

        public FunctionPattern(ExprPattern body, IRArray<ExprPattern> parameters) : this(Utility.GetID(), body, new VArgsPattern(parameters, null))
        {
        }

        public bool MatchLeaf(Function func) => MatchCheckedType(func);
    }
}