using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NnCase
{
    [DebuggerDisplay("{DebuggerDisplay}")]
    public struct ValueRange<T>
    {
        public T Min { get; set; }

        public T Max { get; set; }

        private string DebuggerDisplay => $"{{Min = {Min}, Max = {Max}}}";
    }
}
