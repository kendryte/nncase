using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase
{
    public enum MemoryType
    {
        /// <summary>
        /// Constant memory pool
        /// </summary>
        Constant,

        /// <summary>
        /// Main memory pool
        /// </summary>
        Main,

        /// <summary>
        /// K210 KPU dedicated memory
        /// </summary>
        K210KPU
    }
}
