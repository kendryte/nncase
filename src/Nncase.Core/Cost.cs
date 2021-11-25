using System;
using Nncase.IR;

namespace Nncase
{
    public sealed record Cost(ulong Arith = 0, ulong Memory = 0)
    {
        public Cost(long _Arith, long _Memory) : this((ulong)_Arith, (ulong)_Memory)
        {
            if (_Arith < 0 || _Memory < 0)
            {
                throw new InvalidProgramException("The Cost Can't less than 0!");
            }
        }

        public Cost(long _Arith) : this(_Arith, (long)0) { }

        public ulong Compute()
        {
            return Arith * 2 + Memory;
        }

        public static bool operator >(Cost lhs, Cost rhs) => lhs.Compute() > rhs.Compute();

        public static bool operator <(Cost lhs, Cost rhs) => lhs.Compute() < rhs.Compute();


        private static ulong SafeAdd(ulong lhs, ulong rhs) =>
          (ulong.MaxValue - lhs) > rhs ? lhs + rhs : ulong.MaxValue;

        public static Cost operator +(Cost lhs, Cost rhs) =>
          new Cost(Cost.SafeAdd(lhs.Arith, rhs.Arith),
                  Cost.SafeAdd(lhs.Memory, rhs.Memory));

        public static Cost Inf => new Cost(ulong.MaxValue, ulong.MaxValue);
    }
}