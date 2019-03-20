using System;
using System.Collections.Generic;
using System.Text;
using NnCase.Converter.K210.Converters.Layers;
using NnCase.Converter.K210.Model.Layers;
using NnCase.Converter.Model;
using layers = NnCase.Converter.Model.Layers;

namespace NnCase.Converter.K210.Converters.Stages.Inference
{
    public class InferenceContext
    {
        public int InferenceId { get; set; }

        public Dictionary<Layer, bool> ProcessMap { get; } = new Dictionary<Layer, bool>();

        public KPUMemoryAllocator KPUMemoryAllocator { get; } = new KPUMemoryAllocator();

        public MainMemoryAllocator MainMemoryAllocator { get; } = new MainMemoryAllocator();

        public List<K210Layer> InferenceOrders { get; } = new List<K210Layer>();

        public Dictionary<OutputConnector, MemoryAllocation> KPUMemoryMap { get; } = new Dictionary<OutputConnector, MemoryAllocation>();

        public Dictionary<OutputConnector, MemoryAllocation> MainMemoryMap { get; } = new Dictionary<OutputConnector, MemoryAllocation>();

        public MemoryAllocation GetOrAllocateKPUMemory(OutputConnector output)
        {
            if (!KPUMemoryMap.TryGetValue(output, out var alloc))
            {
                var dimensions = output.Dimensions;
                (var groups, var rowLength, _) = K210Helper.GetRowLayout(dimensions[3]);
                var oneLineChannels = Math.Min(dimensions[1], groups);
                var blocks = (int)Math.Ceiling(dimensions[1] / (double)oneLineChannels);
                if (dimensions[1] == 921)
                {

                }
                var size = rowLength * dimensions[2] * blocks;
                alloc = new MemoryAllocation(KPUMemoryAllocator.Allocate((uint)size));
                KPUMemoryMap.Add(output, alloc);
            }
            else
            {
                alloc.Node.AddRef();
            }

            return alloc;
        }

        public MemoryAllocation GetOrAllocateMainMemory(OutputConnector output)
        {
            if (!MainMemoryMap.TryGetValue(output, out var alloc))
            {
                uint elementSize;
                if (output.Owner.GetType().Name.StartsWith("Quantized"))
                {
                    elementSize = 1;
                }
                else
                {
                    switch (output.Owner)
                    {
                        case K210Conv2d _:
                        case K210AddPadding _:
                        case K210RemovePadding _:
                        case K210Upload _:
                        case layers.Quantize _:
                        case layers.Requantize _:
                            elementSize = 1;
                            break;
                        default:
                            elementSize = 4;
                            break;
                    }
                }

                var dimensions = output.Dimensions;
                alloc = new MemoryAllocation(MainMemoryAllocator.Allocate((uint)dimensions.GetSize() * elementSize));
                MainMemoryMap.Add(output, alloc);
            }
            else
            {
                alloc.Node.AddRef();
            }

            return alloc;
        }
    }
}
