using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.Transform.Pattern;

namespace Nncase.Transform
{
    using ContextEnv = Dictionary<WildCardPattern, ENode>;
    using Tuple = IR.Tuple;

    public sealed class EClassMatcher
    {
        public (bool, ContextEnv) DefaultMatchEnode(ExprPattern pattern, ENode enode, ContextEnv env)
        {
            throw new NotImplementedException($"Unhandled Match Pattern {pattern.GetType()} and Enode {enode.Expr.GetType()} .");
        }


        public (bool, ContextEnv) MatchENodeArgs(IRArray<ExprPattern> Patterns, IRArray<EClass> Childrens, ContextEnv env)
        {
            if (!(Patterns.Count == Childrens.Count))
            {
                return (false, env);
            }
            var new_env = env;
            foreach (var (argPattern, argEclass) in Patterns.Zip(Childrens))
            {
                var (match, looped_env) = MatchEclass(argPattern, argEclass, new_env);
                new_env = looped_env; /* update env */
                if (!match)
                {
                    return (match, env);
                }
            }
            return (true, new_env);
        }

        public (bool, ContextEnv) MatchENode(VarPattern pattern, ENode enode, ContextEnv env)
        {
            if (pattern.MatchLeaf((Var)enode.Expr))
            {
                return (true, env);
            }
            return (false, env);
        }

        public (bool, ContextEnv) MatchENode(ConstPattern pattern, ENode enode, ContextEnv env) => (pattern.MatchLeaf((Const)enode.Expr), env);

        public (bool, ContextEnv) MatchENode(FunctionPattern pattern, ENode enode, ContextEnv env)
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

        public (bool, ContextEnv) MatchENode(CallPattern pattern, ENode enode, ContextEnv env)
        {
            if (pattern.MatchLeaf((Call)enode.Expr))
            {
                return MatchENodeArgs(new[] { pattern.Target }.Concat(pattern.Parameters).ToArray(),
                enode.Children, env);
            }
            return (false, env);
        }
        public (bool, ContextEnv) MatchENode(TuplePattern pattern, ENode enode, ContextEnv env)
        {
            if (pattern.MatchLeaf((Tuple)enode.Expr))
            {
                return MatchENodeArgs(pattern.Fields, enode.Children, env);
            }
            return (false, env);
        }

        public (bool, ContextEnv) MatchENode(OpPattern pattern, ENode enode, ContextEnv env)
        {
            return (pattern.MatchLeaf((Op)enode.Expr), env);
        }

        public (bool, ContextEnv) MatchENode(WildCardPattern pattern, ENode enode, ContextEnv env)
        {
            if (!pattern.MatchLeaf(enode.Expr))
            {
                return (false, env);
            }
            if (!env.ContainsKey(pattern))
            {
                var new_env = new ContextEnv(env);
                new_env.Add(pattern, enode);
                return (true, new_env);
            }
            return (env[pattern] == enode, env);
        }

        public (bool, ContextEnv) MatchENode(ExprPattern pattern, ENode enode, ContextEnv env)
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
                (_, _) => throw new NotImplementedException($"Can't Handle Pattern {pattern.GetType().Name} and Expr {enode.Expr.GetType().Name}")
            };
        }

        public (bool, ContextEnv) MatchEclass(ExprPattern pattern, EClass eclass, ContextEnv env)
        {
            foreach (var enode in eclass.Nodes)
            {
                var (match, new_env) = MatchENode(pattern, enode, env);
                if (match)
                {
                    return (match, new_env);
                }
            }
            return (false, env);
        }

        public static List<(EClass, ContextEnv)> EMatch(EGraph graph, ExprPattern pattern)
        {
            var matcher = new EClassMatcher();
            var matches = new List<(EClass, ContextEnv)>(); // 保存了每个eclassid和入参信息.
            foreach (var eclass in graph.Classes)
            {
                var (match, env) = matcher.MatchEclass(pattern, eclass, new ContextEnv());
                if (match)
                {
                    matches.Add((eclass, env));
                }
            }
            return matches;
        }
    }
}
