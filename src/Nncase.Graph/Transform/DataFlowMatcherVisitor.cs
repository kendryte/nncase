using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Pattern;

namespace Nncase.Transform
{

    using Tuple = IR.Tuple;

    using ContextEnv = Dictionary<ExprPattern, Expr>;

    public record DFMatchResult(Expr Root, ContextEnv Context) : IMatchResult
    {
        public Expr this[ExprPattern expr] => Context[expr];

        public T GetExpr<T>(ExprPattern expr) where T : Expr
        {
            return (T)Context[expr];
        }
        public Expr GetRoot() => Root;
        public T GetRoot<T>() where T : Expr
        {
            return (T)Root;
        }
    }

    public static class DataFlowMatcher
    {
        /// <summary>
        /// Match the Expr with Pattern
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="pattern"></param>
        /// <returns> bool </returns>
        public static List<IMatchResult> Match(Expr expr, ExprPattern pattern)
        {
            if (expr.CheckedType is null) { expr.InferenceType(); }
            var results = new List<IMatchResult>();
            var matcher = new DataFlowMatcherVisitor();
            if (matcher.Visit(pattern, expr))
                results.Add(new DFMatchResult(expr, matcher.Env));
            return results;
        }
    }


    internal sealed class DataFlowMatcherVisitor
    {

        private readonly Dictionary<ExprPattern, bool> _patMemo = new();

        private readonly Dictionary<VArgsPattern, bool> _vargspatMemo = new();


        public readonly ContextEnv Env = new();

        /// <summary>
        /// visit expr
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(ExprPattern pattern, Expr expr)
        {
            return (pattern, expr) switch
            {
                (VarPattern varPat, Var var) => Visit(varPat, var),
                (ConstPattern constPat, Const con) => Visit(constPat, con),
                (FunctionPattern functionPat, Function func) => Visit(functionPat, func),
                (CallPattern callPat, Call call) => Visit(callPat, call),
                (TuplePattern tuplePat, Tuple tuple) => Visit(tuplePat, tuple),
                (OpPattern opPat, Op op) => Visit(opPat, op),
                (WildCardPattern wildCard, _) => Visit(wildCard, expr),
                (OrPattern orPat, _) => Visit(orPat, expr),
                (_, _) => DefaultVisit(pattern, expr)
            };
        }

        /// <summary>
        /// Default visit routine.
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns>Result.</returns>
        public bool DefaultVisit(ExprPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = DefaultVisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

        /// <summary>
        /// Default visit leaf routine.
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns>Result.</returns>
        public bool DefaultVisitLeaf(ExprPattern pattern, Expr expr)
        {
            return false;
        }


        /// <summary>
        /// visit var pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(VarPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }


        /// <summary>
        /// visit call pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(CallPattern pattern, Call expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                Visit(pattern.Target, expr.Target);
                Visit(pattern.Parameters, expr.Parameters);
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

        /// <summary>
        /// visit function pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(ConstPattern pattern, Const expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }


        /// <summary>
        /// visit function pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(FunctionPattern pattern, Function expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                Visit(pattern.Parameters, expr.Parameters);
                Visit(pattern.Body, expr.Body);
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

        /// <summary>
        /// visit op pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(OpPattern pattern, Op expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

        /// <summary>
        /// visit tuple pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(TuplePattern pattern, Tuple expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                Visit(pattern.Fields, expr.Fields);
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }

            return result;
        }

        /// <summary>
        /// visit vargs pattern
        /// </summary>
        /// <param name="patterns"></param>
        /// <param name="exprs"></param>
        /// <returns> bool </returns>
        public bool Visit(VArgsPattern patterns, IRArray<Expr> exprs)
        {
            if (!_vargspatMemo.TryGetValue(patterns, out var result))
            {
                result = VisitLeaf(patterns, exprs);
                _vargspatMemo.Add(patterns, result);
            }
            return result;
        }

        /// <summary>
        ///  visit wildcard pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(WildCardPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

        /// <summary>
        ///  visit or pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool Visit(OrPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                Visit(pattern.Lhs, expr);
                Visit(pattern.Rhs, expr);
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

        /// <summary>
        /// visit var pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(VarPattern pattern, Expr expr)
        {
            if (pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }


        /// <summary>
        /// visit call pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(CallPattern pattern, Call expr)
        {
            if (_patMemo[pattern.Target] &&
               _vargspatMemo[pattern.Parameters] &&
               pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        /// <summary>
        /// visit const pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(ConstPattern pattern, Const expr)
        {
            if (pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        /// <summary>
        /// visit Function pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(FunctionPattern pattern, Function expr)
        {
            if (_patMemo[pattern.Body] &&
              _vargspatMemo[pattern.Parameters] &&
              pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        /// <summary>
        /// visit op pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(OpPattern pattern, Op expr)
        {
            if (pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        /// <summary>
        /// visit tuple pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(TuplePattern pattern, Tuple expr)
        {
            if (pattern.MatchLeaf(expr) &&
             _vargspatMemo[pattern.Fields])
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        /// <summary>
        /// visit Vargs Pattern
        /// </summary>
        /// <param name="patterns"></param>
        /// <param name="exprs"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(VArgsPattern patterns, IRArray<Expr> exprs)
        {
            if (!patterns.MatchLeaf(exprs))
            {
                patterns.MatchEnd(false);
                return false;
            }
            bool result = true;
            foreach (var (i, expr) in (from i in Enumerable.Range(0, exprs.Count)
                                       select (i, exprs[i])))
            {
                result &= Visit(patterns[i], expr);
                if (!result)
                    break;
            }
            patterns.MatchEnd(result);
            return result;
        }

        /// <summary>
        ///  visit wildcard pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(WildCardPattern pattern, Expr expr)
        {
            Env.Add(pattern, expr);
            return true;
        }

        /// <summary>
        ///  visit or pattern
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="expr"></param>
        /// <returns> bool </returns>
        public bool VisitLeaf(OrPattern pattern, Expr expr)
        {
            if (_patMemo[pattern.Lhs])
            {
                Env.Add(pattern, Env[pattern.Lhs]);
            }
            else if (_patMemo[pattern.Rhs])
            {
                Env.Add(pattern, Env[pattern.Rhs]);
            }
            return _patMemo[pattern.Lhs] | _patMemo[pattern.Rhs];
        }
    }
}
