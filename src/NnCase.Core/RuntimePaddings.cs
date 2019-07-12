using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NnCase
{
    [DebuggerDisplay("{DebuggerDisplay}")]
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct RuntimePaddings
    {
        private Padding _d0;
        private Padding _d1;
        private Padding _d2;
        private Padding _d3;

        public Padding this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return _d0;
                    case 1:
                        return _d1;
                    case 2:
                        return _d2;
                    case 3:
                        return _d3;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(index));
                }
            }

            set
            {
                switch (index)
                {
                    case 0:
                        _d0 = value;
                        break;
                    case 1:
                        _d1 = value;
                        break;
                    case 2:
                        _d2 = value;
                        break;
                    case 3:
                        _d3 = value;
                        break;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(index));
                }
            }
        }

        public RuntimePaddings(Padding d0, Padding d1, Padding d2, Padding d3)
        {
            _d0 = d0;
            _d1 = d1;
            _d2 = d2;
            _d3 = d3;
        }

        public Padding[] ToArray()
        {
            return new[] { _d0, _d1, _d2, _d3 };
        }

        private string DebuggerDisplay =>
            $"{{{string.Join(",", ToArray())}}}";
    }
}
