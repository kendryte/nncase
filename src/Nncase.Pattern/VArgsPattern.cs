using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Nncase.IR;
namespace Nncase.Pattern
{

    public abstract record VArgsPattern()
    {
        public virtual ExprPattern this[int index] => this switch
        {
            FixedVArgsPattern fixPat => fixPat[index],
            RepeatVArgsPattern repeatPat => repeatPat[index],
            _ => throw new NotImplementedException($"Can't Handle the Type {this.GetType().Name}!")
        };

        public virtual bool MatchLeaf<T>(IEnumerable<T> other) => this switch
        {
            FixedVArgsPattern fixPat => fixPat.MatchLeaf(other),
            RepeatVArgsPattern repeatPat => repeatPat.MatchLeaf(other),
            _ => throw new NotImplementedException($"Can't Handle the Type {this.GetType().Name}!")
        };

        public virtual void MatchEnd(bool Match)
        {
            switch (this)
            {
                case RepeatVArgsPattern repeatPat:
                    repeatPat.TearDown(Match, repeatPat.Parameters);
                    break;
                default:
                    break;
            }
        }
    }

    public sealed record FixedVArgsPattern(IRArray<ExprPattern> Parameters) : VArgsPattern
    {
        public FixedVArgsPattern(IRArray<Expr> Parameters) : this((from p in Parameters select (ExprPattern)p).ToArray()) { }
        public override ExprPattern this[int index]
        {
            get => Parameters[index];
        }

        public override bool MatchLeaf<T>(IEnumerable<T> other) => Parameters.Count == other.Count();

    }

    public sealed record RepeatVArgsPattern(Action<int, List<ExprPattern>> SetUp, Action<bool, List<ExprPattern>> TearDown) : VArgsPattern
    {
        public readonly List<ExprPattern> Parameters = new();

        public override ExprPattern this[int index]
        {
            get => Parameters[index];
        }

        public override bool MatchLeaf<T>(IEnumerable<T> other)
        {
            if (!Parameters.Any())
                SetUp(other.Count(), Parameters);
            return true;
        }
    }

    public partial class Utility
    {
        public static VArgsPattern IsVArgs(params ExprPattern[] Parameters)
          => new FixedVArgsPattern(Parameters);

        // 默认只生成一次
        public static VArgsPattern IsVArgsRepeat(Action<int, List<ExprPattern>> SetUp)
          => new RepeatVArgsPattern(SetUp, (match, param) =>
          {
              if (match == false)
                  param.Clear();
          });

        public static VArgsPattern IsVArgsRepeat(Action<int, List<ExprPattern>> SetUp, Action<bool, List<ExprPattern>> TearDown)
          => new RepeatVArgsPattern(SetUp, TearDown);
    }
}