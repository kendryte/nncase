// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record VarPattern(string Name, TypePattern Type) : ExprPattern
    {

        public VarPattern(Var var) : this(var.Name, new TypePattern(var.TypeAnnotation)) { }

        private static int _globalVarIndex = 0;

        public VarPattern(TypePattern typePat)
            : this($"var_{_globalVarIndex++}", typePat)
        {
        }

        public VarPattern()
            : this($"var_{_globalVarIndex++}", new TypePattern(AnyType.Default))
        {
        }

        public bool MatchLeaf(Var var)
        {
            return Type.MatchLeaf(var.TypeAnnotation) && MatchCheckedType(var);
        }

    }
}