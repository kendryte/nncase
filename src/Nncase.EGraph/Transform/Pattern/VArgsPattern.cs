using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
namespace Nncase.Transform.Pattern
{
    public sealed record VArgsPattern(IRArray<ExprPattern> Parameters) : ExprPattern, IReadOnlyList<ExprPattern>, IEnumerable<ExprPattern>
    {
        public bool Repeat = false;

        public VArgsPattern(params ExprPattern[] Parameters) : this(ImmutableArray.Create(Parameters)) { }

        public VArgsPattern(IRArray<Expr> Parameters) : this((from p in Parameters select (ExprPattern)p).ToArray()) { }

        public ExprPattern this[int index]
        {
            get
            {
                return Parameters[Repeat ? index % Count : index] switch
                {
                    WildCardPattern wc => Repeat ? wc.Dup(index) : wc,
                    ExprPattern pat => pat
                };
            }
        }

        public int Count => Parameters.Count;

        public bool MatchLeaf<T>(IEnumerable<T> other) => Repeat ? (Count * (other.Count() / Count)) == other.Count() : other.Count() == Count;

        public IEnumerator<ExprPattern> GetEnumerator()
        {
            return Parameters.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)Parameters).GetEnumerator();
        }
    }
}