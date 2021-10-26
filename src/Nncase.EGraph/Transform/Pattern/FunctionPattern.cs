// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record FunctionPattern(string Name, ExprPattern Body, VArgsPattern Parameters) : ExprPattern
    {
        private static int _globalFuncIndex = 0;
        public FunctionPattern(Function func) : this(
            $"func_{_globalFuncIndex++}",
            (ExprPattern)func.Body,
            new VArgsPattern((from p in func.Parameters select ((ExprPattern)p)).ToArray()))
        { }


        public FunctionPattern(ExprPattern body, params ExprPattern[] parameters) : this($"func_{_globalFuncIndex++}", body, new VArgsPattern(parameters))
        {
        }

        public FunctionPattern(ExprPattern body, IRArray<ExprPattern> parameters) : this($"func_{_globalFuncIndex++}", body, new VArgsPattern(parameters))
        {
        }

        public bool MatchLeaf(Function func) => Parameters.MatchLeaf(func.Parameters) && MatchCheckedType(func);

    }
}