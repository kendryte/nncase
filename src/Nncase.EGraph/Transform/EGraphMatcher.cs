using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern;

namespace Nncase.Transform
{
    using EContextEnv = Dictionary<WildCardPattern, ENode>;
    using Tuple = IR.Tuple;
    public record EMatchResult(ENode Root, EContextEnv Context);

    public sealed class EGraphMatcher
    {
        public Dictionary<EClass, List<ENode>> eClasses;

        public EGraphMatcher(Dictionary<EClass, List<ENode>> eclasses)
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
                    return (false, env);
                }
                i++;
            }
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

        public (bool, EContextEnv) MatchENode(WildCardPattern pattern, ENode enode, EContextEnv env)
        {
            if (!pattern.MatchLeaf(enode.Expr))
            {
                return (false, env);
            }
            if (!env.ContainsKey(pattern))
            {
                var new_env = new EContextEnv(env);
                new_env.Add(pattern, enode);
                return (true, new_env);
            }
            return (env[pattern] == enode, env);
        }

        public (bool, EContextEnv) MatchENode(ExprPattern pattern, ENode enode, EContextEnv env)
        {
            return (pattern, enode.Expr) switch
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

        public static List<EMatchResult> EMatch(Dictionary<EClass, List<ENode>> eClasses, params ExprPattern[] patterns)
        {
            var matcher = new EGraphMatcher(eClasses);
            var matchResults = new List<EMatchResult>(); // 保存了每个eclassid和入参信息.
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


        public static List<EMatchResult> EMatch(EGraph eGraph, ExprPattern pattern) => EMatch(eGraph.EClasses(), pattern);
    }
}
