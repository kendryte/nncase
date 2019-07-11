using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NnCase.IR;

namespace NnCase.Runtime
{
    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct MemoryRange
    {
        public MemoryType MemoryType { get; set; }

        public DataType DataType { get; set; }

        public int Start { get; set; }

        public int Size { get; set; }
    }
}
