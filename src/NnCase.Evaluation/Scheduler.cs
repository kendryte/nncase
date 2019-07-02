using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using NnCase.IR;

namespace NnCase.Evaluation
{
    public class MemoryAllocation
    {
        public MemoryType Type { get; set; }

        public int Start { get; set; }

        public int Size { get; set; }

        public int End => Start + Size;

        public bool Overlap(MemoryAllocation allocation)
        {
            return Type == allocation.Type && (Start < allocation.End && End > allocation.Start);
        }
    }

    public class AllocationContext
    {
        private readonly IReadOnlyDictionary<MemoryType, MemoryAllocator> _allocators;
        private readonly Dictionary<OutputConnector, MemoryNode> _memoryMap = new Dictionary<OutputConnector, MemoryNode>();
        private readonly Dictionary<OutputConnector, MemoryAllocation> _allocations = new Dictionary<OutputConnector, MemoryAllocation>();

        public IReadOnlyDictionary<OutputConnector, MemoryAllocation> Allocations => _allocations;

        public AllocationContext(IReadOnlyDictionary<MemoryType, MemoryAllocator> allocators)
        {
            _allocators = allocators;
        }

        public void AllocateDefault(OutputConnector output)
        {
            if (!_allocators.TryGetValue(output.MemoryType, out var allocator))
                throw new InvalidOperationException($"Allocator for ${output.MemoryType} doesn't exist");

            if (_memoryMap.TryGetValue(output, out var memoryNode))
            {
                memoryNode.AddRef();
            }
            else
            {
                var size = allocator.GetBytes(output.Type, output.Shape);
                memoryNode = allocator.Allocate(size);
                _memoryMap.Add(output, memoryNode);
                _allocations.Add(output, new MemoryAllocation { Type = output.MemoryType, Start = memoryNode.Start, Size = size });
            }
        }

        public void Release(OutputConnector output)
        {
            if (_memoryMap.TryGetValue(output, out var memoryNode))
                memoryNode.Release();
        }
    }

    public static class Scheduler
    {
        public static void Schedule(IEnumerable<OutputNode> roots, AllocationContext context, IList<Node> computeSequence)
        {
            var visitor = new RelayDfsVisitor(n =>
            {
                foreach (var output in n.Outputs)
                {
                    foreach (var input in output.Connections)
                    {
                        context.AllocateDefault(output);
                    }
                }

                // check overlap
                {
                    var inputs = new List<MemoryAllocation>();
                    var outputs = new List<MemoryAllocation>();

                    foreach (var output in n.Outputs)
                        outputs.Add(context.Allocations[output]);

                    foreach (var input in n.Inputs)
                    {
                        var alloc = context.Allocations[input.Connection];
                        Debug.Assert(!outputs.Any(x => x.Overlap(alloc)));
                    }
                }

                computeSequence.Add(n);

                // Pin output
                if (!(n is OutputNode))
                {
                    foreach (var input in n.Inputs)
                    {
                        var output = input.Connection;
                        // Pin constant and input
                        if (output.MemoryType != MemoryType.Constant && !(output.Owner is InputNode))
                        {
                            context.Release(output);
                        }
                    }
                }
            });
            visitor.Visit(roots);
        }
    }
}
