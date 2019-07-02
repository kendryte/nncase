using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;

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

        public int End => Start + Size;
    }

    public class MemoryAllocator
    {
        private readonly int _alignment;
        private readonly FreeList _freeList;

        public int MaxUsage => _freeList.MaxUsage;

        public MemoryAllocator(int alignment = 8, int? fixedSpace = null)
        {
            _alignment = alignment;
            _freeList = new FreeList(fixedSpace);
        }

        public MemoryNode Allocate(int size)
        {
            var alignSize = Align(size, _alignment);
            var free = _freeList.Allocate(alignSize);
            var node = new MemoryNode(this, free.Start, free.Size);
            node.AddRef();
            return node;
        }

        public void Free(MemoryNode memoryNode)
        {
            _freeList.Free(new FreeMemoryNode { Start = memoryNode.Start, Size = memoryNode.Size });
        }

        public virtual int GetBytes(DataType dataType, Shape shape)
        {
            return ShapeUtility.GetBytes(dataType) * ShapeUtility.ComputeSize(shape);
        }

        private static int Align(int size, int alignment)
        {
            var rem = size % alignment;
            return rem == 0 ? size : size + (alignment - rem);
        }

        private class FreeList
        {
            private readonly bool _isFixed;
            private readonly SortedList<int, FreeMemoryNode> _freeNodes = new SortedList<int, FreeMemoryNode>();

            public int MaxUsage { get; private set; }

            public FreeList(int? fixedSpace)
            {
                _isFixed = fixedSpace.HasValue;
                if (fixedSpace.HasValue)
                {
                    _freeNodes.Add(0, new FreeMemoryNode { Start = 0, Size = fixedSpace.Value });
                    MaxUsage = fixedSpace.Value;
                }
            }

            public FreeMemoryNode Allocate(int size)
            {
                var node = Reserve(size);
                if (node.Size == size)
                {
                    _freeNodes.Remove(node.Start);
                    return node;
                }
                else
                {
                    node.Size -= size;
                    var newNode = new FreeMemoryNode { Start = node.End, Size = size };
                    return newNode;
                }
            }

            public void Free(FreeMemoryNode node)
            {
                _freeNodes.Add(node.Start, node);
                var index = _freeNodes.IndexOfValue(node);
                Merge(index);
            }

            private FreeMemoryNode Reserve(int size)
            {
                var node = _freeNodes.Values.FirstOrDefault(x => x.Size >= size);
                if (node == null)
                {
                    if (_isFixed)
                        throw new InvalidOperationException("KPU is out of memory");

                    if (_freeNodes.Count != 0)
                    {
                        node = _freeNodes.Values[_freeNodes.Count - 1];
                        if (node.End == MaxUsage)
                        {
                            var enlarge = size - node.Size;
                            node.Size += enlarge;
                            MaxUsage += enlarge;
                            return node;
                        }
                    }
                }

                node = new FreeMemoryNode { Start = MaxUsage, Size = size };
                _freeNodes.Add(MaxUsage, node);
                MaxUsage += size;
                return node;
            }

            private void Merge(int index)
            {
                var node = _freeNodes.Values[index];

                if (index > 0)
                {
                    var leftIndex = index - 1;
                    var left = _freeNodes.Values[leftIndex];
                    if (left.End == node.Start)
                    {
                        left.Size += node.Size;
                        _freeNodes.RemoveAt(index);
                        Merge(leftIndex);
                        return;
                    }
                }

                if (index < _freeNodes.Count - 1)
                {
                    var rightIndex = index + 1;
                    var right = _freeNodes.Values[rightIndex];
                    if (node.End == right.Start)
                    {
                        node.Size += right.Size;
                        _freeNodes.RemoveAt(rightIndex);
                        Merge(index);
                        return;
                    }
                }
            }
        }
    }
}
