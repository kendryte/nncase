// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.Pattern
{
    public sealed record VarPattern(string Name, Func<Var, bool> Cond) : ExprPattern
    {
        public VarPattern(string Name, TypePattern Type) : this(Name, v => Type.MatchLeaf(v.TypeAnnotation)) { }

        public VarPattern(Var var) : this($"var", v => v == var) { }

        public VarPattern()
            : this($"var", new TypePattern(AnyType.Default))
        {
        }

        public VarPattern(TypePattern Type) : this($"var", Type) { }

        public bool MatchLeaf(Var var)
        {
            return Cond(var) && MatchCheckedType(var);
        }
    }

    public static partial class Utility
    {
        public static VarPattern IsVar(TypePattern Type) => new VarPattern(Type);

        public static VarPattern IsVar() => new VarPattern(IsAnyType());
    }
}