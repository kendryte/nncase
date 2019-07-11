using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime
{
    public enum RuntimeOpCode
    {
        Binary = 0,
        Concat = 1,
        Conv2D = 2,
        Dequantize = 3,
    }
}
