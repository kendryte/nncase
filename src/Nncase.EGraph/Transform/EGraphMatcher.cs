using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern;

namespace Nncase.Transform
{
    using EContextEnv = Dictionary<WildCardPattern, ENode>;
    using Tuple = IR.Tuple;
    public record EMatchResult(EClass eClass, EContextEnv Context)
    {
    }

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


        public (bool, EContextEnv) MatchENodeArgs(IRArray<ExprPattern> Patterns, IRArray<EClass> Childrens, EContextEnv env)
        {
            if (!(Patterns.Count == Childrens.Count))
            {
                return (false, env);
            }
            var new_env = env;
            foreach (var (argPattern, argEclass) in Patterns.Zip(Childrens))
            {
                var (match, looped_env) = MatchEclass(argPattern, eClasses[argEclass], new_env);
                new_env = looped_env; /* update env */
                if (!match)
                {
                    return (match, env);
                }
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
                // TODO need check the function Enode's children
                return MatchENodeArgs(
                  pattern.Parameters.Concat(new[] { pattern.Body }).ToArray(),
                  enode.Children, env);
            }
            return (false, env);
        }

        public (bool, EContextEnv) MatchENode(CallPattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Call)enode.Expr))
            {
                return MatchENodeArgs(new[] { pattern.Target }.Concat(pattern.Parameters).ToArray(),
                enode.Children, env);
            }
            return (false, env);
        }
        public (bool, EContextEnv) MatchENode(TuplePattern pattern, ENode enode, EContextEnv env)
        {
            if (pattern.MatchLeaf((Tuple)enode.Expr))
            {
                return MatchENodeArgs(pattern.Fields, enode.Children, env);
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

        public (bool, EContextEnv) MatchEclass(ExprPattern pattern, List<ENode> eNodes, EContextEnv env)
        {
            foreach (var eNode in eNodes)
            {
                var (match, new_env) = MatchENode(pattern, eNode, env);
                if (match)
                {
                    return (match, new_env);
                }
            }
            return (false, env);
        }

        public static List<EMatchResult> EMatch(Dictionary<EClass, List<ENode>> eClasses, ExprPattern pattern)
        {
            var matcher = new EGraphMatcher(eClasses);
            var matchResults = new List<EMatchResult>(); // 保存了每个eclassid和入参信息.
            foreach (var (eclass, enodes) in matcher.eClasses)
            {
                var (match, env) = matcher.MatchEclass(pattern, enodes, new EContextEnv());
                if (match)
                {
                    matchResults.Add(new EMatchResult(eclass, env));
                }
            }
            return matchResults;
        }


        public static List<EMatchResult> EMatch(EGraph eGraph, ExprPattern pattern) => EMatch(eGraph.EClasses(), pattern);
    }
}
