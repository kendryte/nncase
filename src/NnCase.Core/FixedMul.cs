using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase
{
    public struct FixedMul
    {
        public float Mul { get; set; }

        public int Shift { get; set; }

        public int RoundedMul => (int)Math.Round(Mul);
    }
}
