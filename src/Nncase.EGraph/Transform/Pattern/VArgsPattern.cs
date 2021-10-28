using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
namespace Nncase.Transform.Pattern
{
    public sealed record VArgsPattern(IRArray<ExprPattern>? Parameters, Func<int, IRArray<ExprPattern>>? CallBack)
    {
        public IRArray<ExprPattern>? GeneratedParameters { get; set; } = null;

        public VArgsPattern(params ExprPattern[] Parameters) : this(ImmutableArray.Create(Parameters), null) { }

        public VArgsPattern(IRArray<Expr> Parameters) : this((from p in Parameters select (ExprPattern)p).ToArray(), null) { }

        public ExprPattern this[int index]
        {
            get => (Parameters, GeneratedParameters) switch
            {
                (null, IRArray<ExprPattern> parameters) => parameters[index],
                (IRArray<ExprPattern> parameters, null) => parameters[index],
                (_, _) => throw new InvalidOperationException("This VArgsPattern Must have one Parameter!")
            };
        }

        public bool MatchLeaf<T>(IEnumerable<T> other)
        {
            bool createPatterns(Func<int, IRArray<ExprPattern>> callback)
            {
                GeneratedParameters = callback(other.Count());
                return true;
            }

            return (Parameters, CallBack, GeneratedParameters) switch
            {
                (null, not null, null) => createPatterns(CallBack),
                (null, not null, not null) => GeneratedParameters?.Count == other.Count(),
                (not null, null, null) => Parameters?.Count == other.Count(),
                (_, _, _) => false
            };
        }

    }

    public partial class Utility
    {
        public static VArgsPattern IsVArgs(params ExprPattern[] Parameters)
          => new VArgsPattern(Parameters, null);

        public static VArgsPattern IsVArgsRepeat(Func<int, IRArray<ExprPattern>> CallBack)
          => new VArgsPattern(null, CallBack);
    }
}