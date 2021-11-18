using System.Collections.Generic;
using Nncase.Pattern;
using Nncase.IR;
using System;
using System.Linq;

namespace Nncase.Transform
{
    using Tuple = IR.Tuple;

    using EContextEnv = Dictionary<ExprPattern, ENode>;


    public static class EGraphMatcher
    {
        public static List<IMatchResult> Match(Dictionary<EClass, List<ENode>> eClasses, params ExprPattern[] patterns)
        {
            var matcher = new EGraphMatchVisitor(eClasses);
            var matchResults = new List<IMatchResult>();
            foreach (var pattern in patterns)
            {
                foreach (var (eclass, enodes) in matcher.eClasses)
                {
                    var (matchIdx, env) = matcher.MatchEclass(pattern, enodes, new EContextEnv());
                    if (matchIdx != -1)
                    {
                        matchResults.Add(new EMatchResult(enodes[matchIdx], env));
                    }
                }
            }
            return matchResults;
        }

        public static List<IMatchResult> Match(EGraph eGraph, ExprPattern pattern) => Match(eGraph.EClasses(), pattern);

        public static List<IMatchResult> Match(Expr expr, ExprPattern pattern)
        {
            var g = new EGraph();
            g.Add(expr);
            return Match(g, pattern);
        }
    }


    public record EMatchResult(ENode Root, EContextEnv Context) : IMatchResult
    {
        public Expr this[ExprPattern expr] => Context[expr].Expr;

        public T GetExpr<T>(ExprPattern expr) where T : Expr
        {
            return (T)Context[expr].Expr;
        }

        public Expr GetRoot() => Root.Expr;

        public T GetRoot<T>() where T : Expr
        {
            return (T)Root.Expr;
        }
    }

    internal sealed class EGraphMatchVisitor
    {
        public Dictionary<EClass, List<ENode>> eClasses;

        public EGraphMatchVisitor(Dictionary<EClass, List<ENode>> eclasses)
        {
            eClasses = eclasses;
        }

        public (bool, EContextEnv) DefaultMatchEnode(ExprPattern pattern, ENode enode, EContextEnv env)
        {
            throw new NotImplementedException($"Unhandled Match ExprPattern {pattern.GetType()} and Enode {enode.Expr.GetType()} .");
        }


        public (bool, EContextEnv) MatchENode(VArgsPattern Patterns, IEnumerable<EClass> Children, EContextEnv env)
        {
            if (!Patterns.MatchLeaf(Children))
            {
                return (false, env);
            }
            var new_env = env;
            int i = 0;
            foreach (var child in Children)
            {
                var (matchIdx, looped_env) = MatchEclass(Patterns[i], eClasses[child], new_env);
                new_env = looped_env; /* update env */
                if (matchIdx == -1)
                {
                    Patterns.MatchEnd(false);
                    return (false, env);
                }
                i++;
            }
            Patterns.MatchEnd(true);
            return (true, new_env);
        }

        public (bool, EContextEnv) MatchENode(VarPattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Var)enode.Expr))
            {
                return (true, env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(ConstPattern pattern, ENode enode, EContextEnv env) => (pattern.MatchLeaf((Const)enode.Expr), env);

        public (bool, EContextEnv) MatchENode(FunctionPattern pattern, ENode enode, EContextEnv env)
        {
            var func = (Function)enode.Expr;
            if (pattern.MatchLeaf(func))
            {
                var (matchIdx, new_env) = MatchEclass(pattern.Body, eClasses[enode.Children[0]], env);
                if (matchIdx == -1)
                {
                    return (false, env);
                }
                return MatchENode(pattern.Parameters, enode.Children.Skip(1), env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(CallPattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Call)enode.Expr))
            {
                var (matchIdx, new_env) = MatchEclass(pattern.Target, eClasses[enode.Children[0]], env);
                if (matchIdx == -1)
                {
                    return (false, env);
                }
                return MatchENode(pattern.Parameters, enode.Children.Skip(1), new_env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(TuplePattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Tuple)enode.Expr))
            {
                return MatchENode(pattern.Fields, enode.Children, env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(OpPattern pattern, ENode enode, EContextEnv env)
        {
            return (pattern.MatchLeaf((Op)enode.Expr), env);
        }

        private (bool, EContextEnv) UpdateEnv(bool Match, EContextEnv env, ExprPattern pattern, ENode enode)
        {
            if (Match == false)
                return (Match, env);

            if (!env.ContainsKey(pattern))
            {
                var new_env = new EContextEnv(env);
                new_env.Add(pattern, enode);
                return (true, new_env);
            }
            return (env[pattern] == enode, env);
        }

        public (bool, EContextEnv) MatchENode(WildCardPattern pattern, ENode enode, EContextEnv env)
        {
            return (true, env);
        }

        public (bool, EContextEnv) MatchENode(ExprPattern pattern, ENode enode, EContextEnv env)
        {
            var (match, new_env) = (pattern, enode.Expr) switch
            {
                (VarPattern varPat, Var) => MatchENode(varPat, enode, env),
                (ConstPattern conPat, Const) => MatchENode(conPat, enode, env),
                (FunctionPattern functionPat, Function) => MatchENode(functionPat, enode, env),
                (CallPattern callPat, Call) => MatchENode(callPat, enode, env),
                (TuplePattern tuplePat, Tuple) => MatchENode(tuplePat, enode, env),
                (OpPattern opPat, Op) => MatchENode(opPat, enode, env),
                (WildCardPattern wildcardPat, _) => MatchENode(wildcardPat, enode, env),
                (_, _) => (false, env)
            };
            return UpdateEnv(match, new_env, pattern, enode);
        }

        public (int, EContextEnv) MatchEclass(ExprPattern pattern, List<ENode> eNodes, EContextEnv env)
        {
            for (int i = 0; i < eNodes.Count; i++)
            {
                var (match, new_env) = MatchENode(pattern, eNodes[i], env);
                if (match)
                {
                    return (i, new_env);
                }
            }
            return (-1, env);
        }
    }

}