using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace NnCase.Runtime
{
    [StructLayout(LayoutKind.Sequential)]
    public struct NodeHeader
    {
        public RuntimeOpCode OpCode { get; set; }

        public int BodySize { get; set; }
    }
}
