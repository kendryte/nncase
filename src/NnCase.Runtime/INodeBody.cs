using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime
{
    public interface INodeBody
    {
        RuntimeOpCode OpCode { get; }

        void Serialize(BinaryWriter writer);

        void Deserialize(ref MemoryReader reader);
    }
}
