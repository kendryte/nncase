using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NnCase
{
    [DebuggerDisplay("{DebuggerDisplay}")]
    public struct QuantizationParam
    {
        public int ZeroPoint { get; set; }

        public float Scale { get; set; }

        private string DebuggerDisplay => $"{{ZeroPoint = {ZeroPoint}, Scale = {Scale}}}";

        public bool CloseTo(QuantizationParam rhs)
        {
            return ZeroPoint == rhs.ZeroPoint && Math.Abs(Scale - rhs.Scale) <= 1e-7F;
        }
    }
}
