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

        public virtual int Count => this switch
        {
            FixedVArgsPattern fixPat => fixPat.Parameters.Count,
            RepeatVArgsPattern repeatPat => repeatPat.Parameters.Count,
            _ => throw new NotImplementedException($"Can't Handle the Type {this.GetType().Name}!")
        };

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

        public virtual VArgsPattern Copy() => this switch
        {
            FixedVArgsPattern fixedPat => fixedPat.Copy(),
            RepeatVArgsPattern repeatPat => repeatPat.Copy(),
            _ => this with { }
        };

        public virtual void Clear()
        {
            switch (this)
            {
                case (FixedVArgsPattern fixedPat):
                    foreach (var p in fixedPat.Parameters)
                    {
                        p.Clear();
                    }
                    break;
                case (RepeatVArgsPattern repeatPat):
                    repeatPat.MatchEnd(false);
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

        public override VArgsPattern Copy()
        {
            return this with { Parameters = (from p in Parameters select p.Copy()).ToImmutableArray() };
        }
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
            {
                SetUp(other.Count(), Parameters);
                return true;
            }
            return false;
        }

        public override VArgsPattern Copy()
        {
            return this with { };
        }
    }

    public partial class Utility
    {
        public static VArgsPattern IsVArgs(params ExprPattern[] Parameters)
          => new FixedVArgsPattern(Parameters);

        /// <summary>
        /// Create repeated Vargs by template pattern, eg. give the const pattern as Template, will match {Const(),...Const()}
        /// </summary>
        /// <param name="creator">dynamic creator for generator ExprPattern as template</param>
        /// <returns>VArgsPattern</returns>
        public static VArgsPattern IsVArgsRepeat(Func<ExprPattern> creator) => IsVArgsRepeat((n, paramPatterns) =>
        {
            for (int i = 0; i < n; i++)
            {
                paramPatterns.Add(creator());
            }
        });

        /// <summary>
        /// Create repeated Vargs match pattern, it will manual clear the inner container.
        /// </summary>
        /// <param name="SetUp">the int mean matched params nums, list[pattern] is inner params contianer. </param>
        /// <returns>VArgsPattern</returns>
        public static VArgsPattern IsVArgsRepeat(Action<int, List<ExprPattern>> SetUp)
          => new RepeatVArgsPattern(SetUp, (matched, paramPatterns) =>
          {
              if (!matched)
                  paramPatterns.Clear();
          });

        /// <summary>
        /// Create repeated Vargs match pattern
        /// </summary>
        /// <param name="SetUp">the int mean matched params nums, list[pattern] is inner params contianer.</param>
        /// <param name="TearDown">the bool mean current matched state, if match failure your can clear the inner params contianer.</param>
        /// <returns>VArgsPattern</returns>
        public static VArgsPattern IsVArgsRepeat(Action<int, List<ExprPattern>> SetUp, Action<bool, List<ExprPattern>> TearDown)
          => new RepeatVArgsPattern(SetUp, TearDown);
    }
}