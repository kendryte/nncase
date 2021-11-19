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
    }
}