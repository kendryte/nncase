using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NnCase
{
    [DebuggerDisplay("({Before}, {After})")]
    public struct Padding : IEquatable<Padding>
    {
        public int Before { get; set; }

        public int After { get; set; }

        public int Sum => Before + After;

        public static readonly Padding Zero = default;

        public override bool Equals(object obj)
        {
            return obj is Padding padding && Equals(padding);
        }

        public bool Equals(Padding other)
        {
            return Before == other.Before &&
                   After == other.After;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Before, After);
        }

        public static bool operator ==(Padding left, Padding right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Padding left, Padding right)
        {
            return !(left == right);
        }
    }
}
