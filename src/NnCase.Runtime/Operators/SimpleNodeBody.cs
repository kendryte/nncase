using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Runtime.Operators
{
    public abstract class SimpleNodeBody<T> : INodeBody
        where T : unmanaged
    {
        public abstract RuntimeOpCode OpCode { get; }

        public T Options { get; set; }

        public void Deserialize(ref MemoryReader reader)
        {
            Options = reader.Read<T>();
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(Options);
        }
    }
}
