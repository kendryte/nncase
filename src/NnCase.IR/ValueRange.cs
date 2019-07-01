using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.IR
{
    public struct ValueRange<T>
    {
        public T Min { get; set; }

        public T Max { get; set; }
    }
}
