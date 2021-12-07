using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;


namespace Nncase.TIR
{
    public sealed record Select(
      Expr Condition,
Expr TrueValue,
Expr FalseValue
    ) : Expr
    {

    }
}