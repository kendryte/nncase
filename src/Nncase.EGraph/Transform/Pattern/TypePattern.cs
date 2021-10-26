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
    public static partial class Utility
    {
        public static TypePattern IsAnyType() => new TypePattern(AnyType.Default);

        public static TypePattern HasType(Func<IRType, bool> TypeCond) => new TypePattern(TypeCond);

        public static TypePattern HasType(IRType Type) => HasType(x => x == Type);

        public static TypePattern HasDType(Func<DataType, bool> DTypeCond) => new TypePattern(x => x switch
         {
             TensorType ttype => DTypeCond(ttype.DataType),
             _ => false
         });

        public static TypePattern HasDType(DataType DType) => HasDType((DataType x) => x == DType);

        public static TypePattern HasShape(Func<Shape, bool> shapeCond) => new TypePattern(x => x switch
        {
            TensorType ttype => ttype.IsTensor && shapeCond(ttype.Shape),
            _ => false
        });

        public static TypePattern HasShape(Shape shape) => HasShape(x => x == shape);
        
    }

}