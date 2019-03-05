using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace NnCase.Converter.K210.Converters.Stages.Inference
{
    public class MainMemoryAllocator : IMemoryAllocator
    {
        private List<MemoryNode> _nodes = new List<MemoryNode>();

        public uint MaxEnd { get; private set; }

        public MainMemoryAllocator()
        {
        }

        public MemoryNode Allocate(uint size)
        {
            size = Align(size);
            Reserve(size);
            var firstFreeIdx = _nodes.FindLastIndex(o => !o.IsUsed && o.Size >= size);
            Debug.Assert(firstFreeIdx != -1);
            var firstFree = _nodes[firstFreeIdx];
            MemoryNode node;
            if (firstFree.Size == size)
            {
                firstFree.AddRef();
                node = firstFree;
            }
            else
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

            return node;
        }

        private uint Align(uint size)
        {
            var remainder = size % 8;
            if (remainder != 0)
                return size - remainder + 8;
            return size;
        }

        private void Reserve(uint size)
        {
            var firstFreeIdx = _nodes.FindLastIndex(o => !o.IsUsed && o.Size >= size);
            if (firstFreeIdx == -1)
            {
                if (_nodes.Count == 0 || _nodes.Last().IsUsed)
                {
                    _nodes.Add(new MemoryNode(this) { Start = MaxEnd, Size = size });
                    MaxEnd += size;
                }
                else
                {
                    var last = _nodes.Last();
                    var toEnlarge = size - last.Size;
                    last.Size += toEnlarge;
                    MaxEnd += toEnlarge;
                }
            }
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
