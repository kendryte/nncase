using System;
using Nncase.IR;

namespace Nncase
{
    public sealed record Cost(long Arith = 0, long Memory = 0)
    {
        public long Compute()
        {
            return Arith * 2 + Memory;
        }

        public static bool operator >(Cost lhs, Cost rhs) => lhs.Compute() > rhs.Compute();

        public static bool operator <(Cost lhs, Cost rhs) => lhs.Compute() < rhs.Compute();

        public static Cost operator +(Cost lhs, Cost rhs) => new Cost(lhs.Arith + rhs.Arith, lhs.Memory + rhs.Memory);

        public static Cost Inf() => new Cost(long.MaxValue, long.MaxValue);
    }
}