// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public sealed record VarPattern(ID Id, TypePattern Type) : ExprPattern(Id)
    {

        public VarPattern(Var var) : this(Utility.GetID(), new TypePattern(var.TypeAnnotation)) { }

        public VarPattern(TypePattern typePat)
            : this(Utility.GetID(), typePat)
        {
        }

        public VarPattern()
            : this(Utility.GetID(), new TypePattern(AnyType.Default))
        {
        }

        public bool MatchLeaf(Var var)
        {
            return Type.MatchLeaf(var.TypeAnnotation) && MatchCheckedType(var);
        }
    }
    public static partial class Utility
    {
        public static VarPattern IsVar(ID Id, TypePattern Type) => new VarPattern(Id, Type);

        public static VarPattern IsVar(ID Id) => IsVar(Id, IsAnyType());
    }
}