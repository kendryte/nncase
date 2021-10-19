using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Transform.Pattern
{

    public partial record ExprPattern()
    {
        public TypePattern? CheckedTypePat { get; set; }

        public bool MatchCheckedType(Expr expr)
        {
            if (expr.CheckedType is not null && CheckedTypePat is not null)
            {
                return CheckedTypePat.MatchLeaf(expr.CheckedType);
            }
            return true;
        }

        public ExprPattern IsSomeType(Func<IRType, bool> Cond)
        {
            CheckedTypePat = new TypePattern(Cond);
            return this;
        }

        public ExprPattern IsAny() => IsSomeType(x => x == AnyType.Default);

        public ExprPattern IsTensor() => IsSomeType(x => x is TensorType);

        public ExprPattern IsScalar() => IsSomeType(x => x switch
             {
                 TensorType xt => xt.IsScalar,
                 _ => false
             });
    };

}