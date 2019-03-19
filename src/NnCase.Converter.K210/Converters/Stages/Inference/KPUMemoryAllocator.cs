using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Inference
{
    public class KPUMemoryAllocator : IMemoryAllocator
    {
        private List<MemoryNode> _nodes = new List<MemoryNode>();

        public uint MaxStart { get; private set; } = 2 * 1024 * 1024 / 64;

        public uint MaxUsage => 2 * 1024 * 1024 / 64 - MaxStart;

        public bool _findFromLast = true;

        public KPUMemoryAllocator()
        {
            _nodes.Add(new MemoryNode(this) { Start = 0, Size = MaxStart });
        }

        public MemoryNode Allocate(uint size)
        {
            var firstFreeIdx = _nodes.FindLastIndex(o => !o.IsUsed && o.Size >= size);
            if (firstFreeIdx == -1)
                throw new InvalidOperationException("KPU ran out of memory.");
            var firstFree = _nodes[firstFreeIdx];
            MemoryNode node;
            if (firstFree.Size == size)
            {
                firstFree.AddRef();
                node = firstFree;
            }
            else
            {
                if (_findFromLast)
                {
                    firstFree.Size -= size;
                    var newNode = new MemoryNode(this)
                    {
                        Start = firstFree.Start + firstFree.Size,
                        Size = size
                    };
                    newNode.AddRef();

                    _nodes.Insert(firstFreeIdx + 1, newNode);
                    node = newNode;
                }
                else
                {
                    var newNode = new MemoryNode(this)
                    {
                        Start = firstFree.Start,
                        Size = size
                    };
                    newNode.AddRef();

                    firstFree.Start += size;
                    firstFree.Size -= size;
                    _nodes.Insert(firstFreeIdx, newNode);
                    node = newNode;
                }

                _findFromLast = !_findFromLast;
            }

            MaxStart = Math.Min(node.Start, MaxStart);
            return node;
        }

        public void Free(MemoryNode node)
        {
            Debug.Assert(!node.IsUsed);
            var idx = _nodes.IndexOf(node);
            if (idx != 0)
            {
                var before = _nodes[idx - 1];
                if (!before.IsUsed)
                {
                    before.Size += node.Size;
                    _nodes.RemoveAt(idx);
                    idx--;
                    node = _nodes[idx];
                }
            }
            if (idx != _nodes.Count - 1)
            {
                var after = _nodes[idx + 1];
                if (!after.IsUsed)
                {
                    node.Size += after.Size;
                    _nodes.RemoveAt(idx + 1);
                }
            }
        }
    }
}
