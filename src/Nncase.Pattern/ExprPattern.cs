using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.Pattern
{
    /// <summary>
    /// Base Record For Pattern
    /// </summary>
    /// <param name="Id"></param>
    public abstract partial record ExprPattern(int Id)
    {
        private static int _globalPatIndex = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="ExprPattern"/> class.
        /// </summary>
        public ExprPattern() : this(_globalPatIndex++) { }

        /// <summary>
        /// Convert Expr to Pattern
        /// </summary>
        /// <param name="expr"></param>
        public static implicit operator ExprPattern(Expr expr) => expr switch
        {
            (Var var) => new VarPattern(var),
            (Const con) => new ConstPattern(con),
            (Function function) => new FunctionPattern(function),
            (Call call) => new CallPattern(call),
            (IR.Tuple tuple) => new TuplePattern(tuple),
            (Op op) => OpPattern.CastToPattern(op),
            _ => throw new NotImplementedException($"Can't Convert The Expr {expr.GetType().Name} To ExprPattern")
        };

        /// <summary>
        /// Pattern for CheckedType
        /// </summary>
        public TypePattern? CheckedTypePat { get; set; }

        /// <summary>
        /// Match The Expr Type
        /// </summary>
        /// <param name="expr"></param>
        /// <returns> is Matched </returns>
        public bool MatchCheckedType(Expr expr) => (expr.CheckedType, this.CheckedTypePat) switch
        {
            (null, null) => true,
            (null, TypePattern pat) => false,
            (IRType type, null) => true,
            (IRType type, TypePattern pat) => pat.MatchLeaf(type)
        };


        /// <summary>
        /// Match The Expr
        /// </summary>
        /// <param name="expr"></param>
        /// <returns> is matched </returns>
        public virtual bool MatchLeaf(Expr expr) => (this, expr) switch
        {
            (VarPattern varPat, Var var) => varPat.MatchLeaf(var),
            (ConstPattern conPat, Const con) => conPat.MatchLeaf(con),
            (FunctionPattern functionPat, Function function) => functionPat.MatchLeaf(function),
            (CallPattern callPat, Call call) => callPat.MatchLeaf(call),
            (TuplePattern tuplePat, IR.Tuple tuple) => tuplePat.MatchLeaf(tuple),
            (OpPattern opPat, Op op) => opPat.MatchLeaf(op),
            (WildCardPattern wildcardPat, _) => wildcardPat.MatchLeaf(expr),
            (_, _) => false
        };

        /// <summary>
        /// Add the Type Patten For Current Pattern
        /// </summary>
        /// <param name="Cond"></param>
        /// <returns> this </returns>
        public ExprPattern IsSomeType(Func<IRType, bool> Cond)
        {
            CheckedTypePat = new TypePattern(Cond);
            return this;
        }

        /// <summary>
        /// Add type Pattern
        /// </summary>
        /// <returns> this </returns>
        public ExprPattern IsAny() => IsSomeType(x => x == AnyType.Default);

        /// <summary>
        /// Add type Pattern
        /// </summary>
        /// <returns> this </returns>
        public ExprPattern IsTensor() => IsSomeType(x => x is TensorType);

        /// <summary>
        /// Add type Pattern
        /// </summary>
        /// <returns> this </returns>
        public ExprPattern IsScalar() => IsSomeType(x => x switch
                     {

                         TensorType xt => xt.IsScalar,
                         _ => false
                     });

        /// <summary>
        /// Copy The **New** ExprPattern. NOTE the new pattern have different Id with old one, The there not equal.
        /// </summary>
        /// <returns> ExprPattern </returns>
        public ExprPattern Copy() => this with { Id = _globalPatIndex++ };
    };

    public static partial class Utility
    {
        /// <summary>
        /// Get the current expr checked Shape
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static List<Dimension> GetShape(Expr expr) => expr.CheckedType switch
        {
            TensorType type => new List<Dimension>(type.Shape),
            _ => throw new InvalidOperationException($"The Expr {expr.GetType().Name} Has No Shape!")
        };
    }

}