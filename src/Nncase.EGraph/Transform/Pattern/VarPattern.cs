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

        public VarPattern(Var var) : this($"var", new TypePattern(var.TypeAnnotation)) { }

        public VarPattern()
            : this($"var", new TypePattern(AnyType.Default))
        {
        }

        public VarPattern(TypePattern Type) : this($"var", Type) { }

        public bool MatchLeaf(Var var)
        {
            return Type.MatchLeaf(var.TypeAnnotation) && MatchCheckedType(var);
        }
    }
    public static partial class Utility
    {
        public static VarPattern IsVar(TypePattern Type) => new VarPattern(Type);

        public static VarPattern IsVar() => new VarPattern(IsAnyType());
    }
}