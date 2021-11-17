using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Pattern;

namespace Nncase.Transform
{

    using Tuple = IR.Tuple;

    using ContextEnv = Dictionary<ExprPattern, Expr>;

    public record MatchResult(Expr Root, ContextEnv Context) : IMatchResult
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


    internal sealed class DataFlowMatcher
    {

        private readonly Dictionary<ExprPattern, bool> _patMemo = new();
        private readonly Dictionary<VArgsPattern, bool> _vargspatMemo = new();

        public readonly ContextEnv Env = new();

        public static (bool, ContextEnv) Match(Expr expr, ExprPattern pattern)
        {
            var matcher = new DataFlowMatcher();
            return (matcher.Visit(pattern, expr), matcher.Env);
        }

        public bool Visit(ExprPattern pattern, Expr expr)
        {
            return (pattern, expr) switch
            {
                (VarPattern varPat, Var var) => varPat.MatchLeaf(var) ? Visit(varPat, var) : false,
                (ConstPattern constPat, Const con) => constPat.MatchLeaf(con) ? Visit(constPat, con) : false,
                (FunctionPattern functionPat, Function func) => functionPat.MatchLeaf(func) ? Visit(functionPat, func) : false,
                (CallPattern callPat, Call call) => callPat.MatchLeaf(call) ? Visit(callPat, call) : false,
                (TuplePattern tuplePat, Tuple tuple) => tuplePat.MatchLeaf(tuple) ? Visit(tuplePat, tuple) : false,
                (OpPattern opPat, Op op) => opPat.MatchLeaf(op) ? Visit(opPat, op) : false,
                (WildCardPattern wildCard, _) => wildCard.MatchLeaf(expr) ? Visit(wildCard, expr) : false,
                (_, _) => false
            };
        }

        public bool Visit(VarPattern pattern, Expr expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(expr, result);
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
                _patMemo.Add(expr, result);
            }
            return result;
        }

        public bool Visit(ConstPattern pattern, Const expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(expr, result);
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
                _patMemo.Add(expr, result);
            }
            return result;
        }

        public bool Visit(OpPattern pattern, Op expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(expr, result);
            }
            return result;
        }

        public bool Visit(TuplePattern pattern, Tuple expr)
        {
            if (!_patMemo.TryGetValue(pattern, out var result))
            {
                Visit(pattern.Fields, expr.Fields);
                result = VisitLeaf(pattern, expr);
                _patMemo.Add(expr, result);
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


        public bool VisitLeaf(VarPattern pattern, Expr expr)
        {
            Env.Add(pattern, expr);
            return true;
        }

        public bool VisitLeaf(CallPattern pattern, Call expr)
        {
            if (_patMemo[pattern.Target] &&
               _vargspatMemo[pattern.Parameters])
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        public bool VisitLeaf(ConstPattern pattern, Const expr)
        {
            Env.Add(pattern, expr);
            return true;
        }

        public bool VisitLeaf(FunctionPattern pattern, Function expr)
        {
            if (_patMemo[pattern.Body] &&
              _vargspatMemo[pattern.Parameters])
            {
                Env.Add(pattern, expr);
                return true;
            }
            return false;
        }

        public bool VisitLeaf(OpPattern pattern, Op expr)
        {
            Env.Add(pattern, expr);
            return true;
        }

        public bool VisitLeaf(TuplePattern pattern, Tuple expr)
        {
            if (_vargspatMemo[pattern.Fields])
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
            if (result)
                _vargspatMemo.Add(patterns, result);
            return result;
        }
    }

    // internal sealed class DataFlowMutator : ExprVisitor<Expr, bool>
    // {

    //     public ExprPattern Pattern { get; set; } = IsWildCard();

    //     public static Expr ReWrite(Expr pre, PatternRule[] rules)
    //     {
    //         var post = pre;
    //         var last = post;
    //         var mutator = new DataFlowMutator();
    //         var equal = true;
    //         do
    //         {
    //             equal = true;
    //             foreach (var rule in rules)
    //             {
    //                 foreach (var pattern in rule.Patterns)
    //                 {
    //                     last = post;
    //                     mutator.Pattern = pattern;
    //                     post = mutator.Visit(last);
    //                     equal = post == last;
    //                     if (!equal)
    //                         break;
    //                 }
    //                 if (!equal)
    //                     break;
    //             }
    //         } while (!equal);
    //         return post;
    //     }

    //     public override Expr VisitLeaf(Expr expr)
    //     {
    //         return (Pattern, expr) switch
    //         {
    //             (VarPattern varPat, Var var) => VisitLeaf(var),
    //             (ConstPattern constPat, Const con) => VisitLeaf(con),
    //             (FunctionPattern functionPat, Function func) => VisitLeaf(func),
    //             (CallPattern callPat, Call call) => VisitLeaf(call),
    //             (TuplePattern tuplePat, Tuple tuple) => VisitLeaf(tuple),
    //             (OpPattern opPat, Op op) => VisitLeaf(op),
    //             _ => DefaultVisitLeaf(expr),
    //         };
    //     }
    //     // public override Expr VisitLeaf(Const expr)
    //     // {
    //     //     ((ConstPattern)Pattern).MatchLeaf(expr);
    //     //     // if ()
    //     // }
    // }
}
