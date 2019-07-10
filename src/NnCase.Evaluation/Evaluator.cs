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

        public Memory<byte> MemoryAt(MemoryAllocation allocation)
        {
            var memoryPool = _memoryPools[allocation.Type];
            var memory = memoryPool.Slice(allocation.Start, allocation.Size);
            return memory;
        }

        public Memory<byte> MemoryAt(OutputConnector connector)
        {
            return MemoryAt(_allocations[connector]);
        }

        public Memory<byte> MemoryAt(InputConnector connector)
        {
            return MemoryAt(_allocations[connector.Connection]);
        }

        public Memory<byte> InputAt(int index)
        {
            return MemoryAt(_inputs[index]);
        }

        public Memory<byte> OutputAt(int index)
        {
            return MemoryAt(_outputs[index]);
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

        public Span<T> MemoryAt<T>(InputConnector connector)
            where T : unmanaged
        {
            return MemoryAt<T>(_allocations[connector.Connection]);
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

        public void Evaluate(Quantizer quantizer = null, bool dumpDuration = false)
        {
            var stopwatch = new Stopwatch();
            var totalDuration = TimeSpan.Zero;

            foreach (var node in _computeSequence)
            {
                stopwatch.Restart();
                if (!_evaluatorRegistry.TryInvoke(node, this))
                    throw new NotSupportedException($"Evaluator for {node.GetType().Name} is not found");
                stopwatch.Stop();

                var duration = stopwatch.Elapsed;
                totalDuration += duration;
                if (dumpDuration)
                    Console.WriteLine($"{node.GetType().Name}: {duration.TotalMilliseconds:F2} ms");

                if (quantizer != null)
                {
                    if (node is FakeQuantize || node is FakeDequantize)
                    {
                        var output = node.Outputs[0];
                        quantizer.Record(output, MemoryAt<float>(output));
                    }
                }
            }

            if (dumpDuration)
                Console.WriteLine($"Total: {totalDuration.TotalMilliseconds:F2} ms");
        }

        private void InitializeConstant(Constant constant)
        {
            var mem = MemoryAt<byte>(constant.Output);
            constant.Data.Span.CopyTo(mem);
        }
    }
}
