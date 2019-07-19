using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Evaluation;
using NnCase.IR;

namespace NnCase.Targets.K210.Evaluation
{
    internal class KPUMemoryAllocator : MemoryAllocator
    {
        public KPUMemoryAllocator()
            : base(K210Helper.KPUMemoryLineSize, K210Helper.KPUMemorySize)
        {
        }

        public override int GetBytes(DataType dataType, Shape shape)
        {
            if (dataType != DataType.UInt8)
                throw new ArgumentException("Invalid dataty[e");
            return K210Helper.GetBytes(shape);
        }
    }
}
