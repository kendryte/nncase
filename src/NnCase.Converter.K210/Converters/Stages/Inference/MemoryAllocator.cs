using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Inference
{
    public interface IMemoryAllocator
    {
        void Free(MemoryNode node);
    }

    public class MemoryNode
    {
        private readonly IMemoryAllocator _memoryAllocator;
        private int _useCount;

        public uint Start { get; set; }

        public uint ValidStart
        {
            get
            {
                if (IsUsed)
                    return Start;
                else
                    throw new InvalidOperationException("Memory node has been free.");
            }
        }

        public uint Size { get; set; }

        public bool IsUsed => _useCount != 0;

        public MemoryNode(IMemoryAllocator memoryAllocator)
        {
            _memoryAllocator = memoryAllocator;
        }

        public void AddRef()
        {
            _useCount++;
        }

        public void Release()
        {
            if (--_useCount == 0)
            {
                _memoryAllocator.Free(this);
            }

            Debug.Assert(_useCount >= 0);
        }
    }

    public class MemoryAllocation
    {
        public MemoryNode Node { get; set; }

        public uint Offset { get; set; }

        public uint Size { get; set; }

        public uint GetAddress() => Node.ValidStart + Offset;

        public MemoryAllocation(MemoryNode memoryNode)
        {
            Node = memoryNode;
            Size = memoryNode.Size;
        }

        public MemoryAllocation(MemoryNode memoryNode, uint offset, uint size)
        {
            Node = memoryNode;
            Offset = offset;
            Size = size;

            if (Offset + Size > Node.Size)
                throw new ArgumentOutOfRangeException(nameof(offset));
        }
    }
}
