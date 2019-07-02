using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Evaluation
{
    public class Evaluator
    {
        private readonly IReadOnlyDictionary<MemoryType, MemoryAllocator> _allocators;
        private readonly IReadOnlyDictionary<OutputConnector, MemoryAllocation> _allocations;
        private readonly IReadOnlyList<Node> _computeSequence;
        private readonly EvaluatorRegistry _evaluatorRegistry;
        private readonly List<MemoryAllocation> _inputs = new List<MemoryAllocation>();
        private readonly List<MemoryAllocation> _outputs = new List<MemoryAllocation>();
        private readonly Dictionary<MemoryType, Memory<byte>> _memoryPools = new Dictionary<MemoryType, Memory<byte>>();

        public Evaluator(IReadOnlyDictionary<MemoryType, MemoryAllocator> allocators, IReadOnlyDictionary<OutputConnector, MemoryAllocation> allocations, IReadOnlyList<Node> computeSequence, EvaluatorRegistry evaluatorRegistry)
        {
            _allocators = allocators;
            _allocations = allocations;
            _computeSequence = computeSequence;
            _evaluatorRegistry = evaluatorRegistry;
            foreach (var allocator in _allocators)
                _memoryPools.Add(allocator.Key, new byte[allocator.Value.MaxUsage]);

            foreach (var node in _computeSequence)
            {
                switch (node)
                {
                    case InputNode i:
                        _inputs.Add(_allocations[i.Output]);
                        break;
                    case OutputNode o:
                        _outputs.Add(_allocations[o.Input.Connection]);
                        break;
                    case Constant c:
                        InitializeConstant(c);
                        break;
                }
            }
        }

        public Span<T> MemoryAt<T>(MemoryAllocation allocation)
            where T : unmanaged
        {
            var memoryPool = _memoryPools[allocation.Type];
            var span = memoryPool.Span.Slice(allocation.Start, allocation.Size);
            return MemoryMarshal.Cast<byte, T>(span);
        }

        public Span<T> MemoryAt<T>(OutputConnector connector)
            where T : unmanaged
        {
            return MemoryAt<T>(_allocations[connector]);
        }

        public Span<T> InputAt<T>(int index)
            where T : unmanaged
        {
            return MemoryAt<T>(_inputs[index]);
        }

        public Span<T> OutputAt<T>(int index)
            where T : unmanaged
        {
            return MemoryAt<T>(_outputs[index]);
        }

        public void Evaluate()
        {
            var stopwatch = new Stopwatch();

            foreach (var node in _computeSequence)
            {
                stopwatch.Restart();
                if (!_evaluatorRegistry.TryInvoke(node, this))
                    throw new NotImplementedException($"Evaluator for {node.GetType().Name} is not implemented");
                stopwatch.Stop();

                var duration = stopwatch.Elapsed;
                Console.WriteLine($"{node.GetType().Name}: {duration.TotalMilliseconds} ms");
            }
        }

        private void InitializeConstant(Constant constant)
        {
            var mem = MemoryAt<byte>(constant.Output);
            constant.Data.Span.CopyTo(mem);
        }
    }
}
