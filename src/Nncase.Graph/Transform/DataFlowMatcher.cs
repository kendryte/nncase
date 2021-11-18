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


    public sealed class DataFlowMatcher
    {

        private readonly Dictionary<ExprPattern, bool> _patMemo = new();
        private readonly Dictionary<VArgsPattern, bool> _vargspatMemo = new();

        public readonly ContextEnv Env = new();

        public static List<IMatchResult> Match(Expr expr, ExprPattern pattern)
        {
            var results = new List<IMatchResult>();
            var matcher = new DataFlowMatcher();
            if (matcher.Visit(pattern, expr))
                results.Add(new DFMatchResult(expr, matcher.Env));
            return results;
        }

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
                (_, _) => false
            };
        }

        public bool Visit(VarPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

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

        public bool Visit(ConstPattern pattern, Const expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }


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

        public bool Visit(OpPattern pattern, Op expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }

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

        public bool Visit(VArgsPattern patterns, IRArray<Expr> exprs)
        {
            if (!_vargspatMemo.TryGetValue(patterns, out var result))
            {
                result = VisitLeaf(patterns, exprs);
                _vargspatMemo.Add(patterns, result);
            }
            return result;
        }

        public bool Visit(WildCardPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(pattern, result);
            }
            return result;
        }


        public bool VisitLeaf(VarPattern pattern, Expr expr)
        {
            if (pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

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

        public bool VisitLeaf(ConstPattern pattern, Const expr)
        {
            if (pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

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

        public bool VisitLeaf(OpPattern pattern, Op expr)
        {
            if (pattern.MatchLeaf(expr))
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

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

        public bool VisitLeaf(VArgsPattern patterns, IRArray<Expr> exprs)
        {
            if (!patterns.MatchLeaf(exprs))
                return false;
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

        public bool VisitLeaf(WildCardPattern pattern, Expr expr)
        {
            Env.Add(pattern, expr);
            return true;
        }
    }
}
