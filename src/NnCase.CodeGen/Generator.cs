using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using NnCase.Evaluation;
using NnCase.Evaluation.Operators;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Runtime;
using BinaryWriter = NnCase.Runtime.BinaryWriter;

namespace NnCase.CodeGen
{
    public class Generator
    {
        private readonly IReadOnlyDictionary<MemoryType, MemoryAllocator> _allocators;
        private readonly IReadOnlyDictionary<OutputConnector, MemoryAllocation> _allocations;
        private readonly IReadOnlyList<Node> _computeSequence;
        private readonly CodeGenRegistry _codeGenRegistry;
        private readonly List<RuntimeShape> _inputShapes = new List<RuntimeShape>();
        private readonly List<MemoryRange> _inputs = new List<MemoryRange>();
        private readonly List<MemoryRange> _outputs = new List<MemoryRange>();
        private readonly Memory<byte> _constants;
        private int _nodesCount;

        public Generator(IReadOnlyDictionary<MemoryType, MemoryAllocator> allocators, IReadOnlyDictionary<OutputConnector, MemoryAllocation> allocations, IReadOnlyList<Node> computeSequence, CodeGenRegistry codeGenRegistry)
        {
            _allocators = allocators;
            _allocations = allocations;
            _computeSequence = computeSequence;
            _codeGenRegistry = codeGenRegistry;
            _constants = new byte[allocators[MemoryType.Constant].MaxUsage];
            _nodesCount = _computeSequence.Count(x => _codeGenRegistry.HasRuntime(x.GetType()));

            foreach (var node in _computeSequence)
            {
                switch (node)
                {
                    case InputNode i:
                        _inputs.Add(MemoryRange(i.Output));
                        _inputShapes.Add(OpUtility.To(i.Output.Shape));
                        break;
                    case OutputNode o:
                        _outputs.Add(MemoryRange(o.Input.Connection));
                        break;
                    case Constant c:
                        InitializeConstant(c);
                        break;
                }
            }
        }

        public void Generate(Stream output)
        {
            using (var writer = new BinaryWriter(output))
            {
                WriteHeader(writer);
                var nodeHeadersPos = WriteNodeHeaders(writer);
                var nodeHeaders= WriteNodes(writer);
                FixNodeHeaders(writer, nodeHeadersPos, nodeHeaders);
            }
        }

        private void WriteHeader(BinaryWriter writer)
        {
            var header = new ModelHeader
            {
                Identifier = ModelHeader.IdentifierValue,
                Version = ModelHeader.CurrentVersion,
                Flags = 0,
                Target = TargetId.K210,
                ConstantUsage = _constants.Length,
                MainMemoryUsage = _allocators[MemoryType.Main].MaxUsage,
                Nodes = _nodesCount,
                Inputs = _inputs.Count,
                Outputs = _outputs.Count
            };

            writer.Write(header);
        }

        private long WriteNodeHeaders(BinaryWriter writer)
        {
            // inputs
            writer.Write(_inputs);
            writer.Write(_inputShapes);
            // outputs
            writer.Write(_outputs);
            // nodes
            var nodeHeadersPos = writer.Position;
            var reserved = Unsafe.SizeOf<NodeHeader>() * _nodesCount;
            writer.Position += reserved;
            return nodeHeadersPos;
        }

        private List<NodeHeader> WriteNodes(BinaryWriter writer)
        {
            var nodeHeaders = new List<NodeHeader>();

            foreach (var node in _computeSequence)
            {
                if (!_codeGenRegistry.TryInvoke(node, this, out var body))
                    throw new NotSupportedException($"Emitter for {node.GetType().Name} is not found");

                if (body != null)
                {
                    var start = writer.Position;
                    body.Serialize(writer);
                    writer.AlignPosition(8);
                    var size = writer.Position - start;
                    nodeHeaders.Add(new NodeHeader { OpCode = body.OpCode, BodySize = (int)size });
                }
            }

            return nodeHeaders;
        }

        private void FixNodeHeaders(BinaryWriter writer, long nodeHeadersPos, List<NodeHeader> nodeHeaders)
        {
            var endPos = writer.Position;
            writer.Position = nodeHeadersPos;
            writer.Write(nodeHeaders);
            writer.Position = endPos;
        }

        private void InitializeConstant(Constant constant)
        {
            var allocation = _allocations[constant.Output];
            constant.Data.CopyTo(_constants.Slice(allocation.Start, allocation.Size));
        }

        public MemoryRange MemoryRange(OutputConnector connector)
        {
            var allocation = _allocations[connector];
            return new MemoryRange
            {
                MemoryType = connector.MemoryType,
                DataType = connector.Type,
                Start = allocation.Start,
                Size = allocation.Size
            };
        }

        public MemoryRange MemoryRange(InputConnector connector)
        {
            return MemoryRange(connector.Connection);
        }
    }
}
