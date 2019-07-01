using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Evaluation
{
    public sealed class MemoryNode
    {
        private readonly MemoryAllocator _memoryAllocator;

        public int Start { get; }

        public int Size { get; }

        public int End => Start + Size;

        public int UseCount { get; private set; }

        public bool IsUsed => UseCount > 0;

        public int SafeStart
        {
            get
            {
                if (!IsUsed)
                    throw new ObjectDisposedException("memoryNode");

                return Start;
            }
        }

        public MemoryNode(MemoryAllocator memoryAllocator, int start, int size)
        {
            _memoryAllocator = memoryAllocator;
            Start = start;
            Size = size;
            UseCount = 0;
        }

        public void AddRef()
        {
            ++UseCount;
        }

        public void Release()
        {
            if (--UseCount == 0)
                _memoryAllocator.Free(this);

            if (UseCount < 0)
                throw new ObjectDisposedException("memoryNode");
        }
    }

    public sealed class FreeMemoryNode
    {
        public int Start { get; set; }

        public int Size { get; set; }
    }

    public class MemoryAllocator
    {
        public MemoryNode Allocate(int size)
        {
            throw new NotImplementedException();
        }

        public void Free(MemoryNode memoryNode)
        {
            throw new NotImplementedException();
        }
    }
}
