using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{
    public record TypePattern(Func<IRType, bool> Cond)
    {
        public TypePattern(IRType ValueType) : this(x => (x == ValueType)) { }
        public TypePattern(AnyType ValueType) : this(x => (x == ValueType)) { }
        public TypePattern(TensorType ValueType) : this(x => (x == ValueType)) { }
        public TypePattern(InvalidType ValueType) : this(x => (x == ValueType)) { }
        public TypePattern(TupleType ValueType) : this(x => (x == ValueType)) { }
        public TypePattern(CallableType ValueType) : this(x => (x == ValueType)) { }
        public bool MatchLeaf(IRType ValueType)
        {
            return Cond(ValueType);
        }
    }
    public static partial class Functional
    {
        public static TypePattern IsAnyType() => new TypePattern(AnyType.Default);

    }

}