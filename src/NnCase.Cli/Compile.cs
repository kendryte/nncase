using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.Evaluation;
using NnCase.Evaluation.Data;
using NnCase.Importer;
using NnCase.IR;
using NnCase.Targets;
using NnCase.Transforms;

namespace NnCase.Cli
{
    public class Compile
    {
        private readonly IServiceProvider _serviceProvider;

        public Compile(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public async Task Run(Options options)
        {
            // 1. Import
            var graph = await ImportGraph(options);
            var target = new Targets.CPU.CPUTarget();
            target.RegisterEvaluators(EvaluatorRegistry.Default);

            // 2. Optimize Pass 1 (Generic)
            OptimizePass1(graph);

            // 3. Optimize Pass 2 (Target aware)
            OptimizePass2(graph, target);

            // 4. Quantize
            await Quantize(options, graph);
        }

        private async Task<Graph> ImportGraph(Options options)
        {
            var model = await File.ReadAllBytesAsync(options.Input);
            var graph = new Graph();

            switch (options.InputFormat)
            {
                case "tflite":
                    new TFLiteImporter(_serviceProvider, model, graph).Import();
                    break;
                default:
                    throw new ArgumentException($"Unsupported input format: {options.InputFormat}");
            }

            DumpGraph(graph, "import");
            return graph;
        }

        private void OptimizePass1(Graph graph)
        {
            var transforms = new Transform[]
            {
                new FoldTransposeTransform(),
                new FoldNopTransposeTransform(),
                new FoldNopReshapeTransform(),
                new ConstantFoldingTransform(),
                new TransposeBinaryMotionTransform(),
                new TransposeConcatMotionTransform(),
                new TransposeReduceMotionTransform()
            };

            Transform.TransformGraph(graph, transforms);
            DumpGraph(graph, "optimize_1");
        }

        private void DumpGraph(Graph graph, string name)
        {
            using (var stream = File.Create($"ir_{name}.dot"))
            {
                graph.DumpDotGraph(stream);
            }
        }

        private void OptimizePass2(Graph graph, Target target)
        {
            target.OptimizePass2(graph);
            DumpGraph(graph, "optimize_2");
        }

        private async Task Quantize(Options options, Graph graph)
        {
            // 3.1. Add quantization checkpoints
            AddQuantizationCheckpoints(graph);

            // 3.2 Get activation ranges
            await GetActivationRanges(options, graph);
        }

        private void AddQuantizationCheckpoints(Graph graph)
        {
        }

        private async Task GetActivationRanges(Options options, Graph graph)
        {
            var allocators = new Dictionary<MemoryType, MemoryAllocator>
            {
                { MemoryType.Constant, new MemoryAllocator() },
                { MemoryType.Main, new MemoryAllocator() }
            };
            var allocationContext = new AllocationContext(allocators);
            var computeSequence = new List<Node>();
            Scheduler.Schedule(graph.Outputs, allocationContext, computeSequence);

            var evaluator = new Evaluator(allocators, allocationContext.Allocations, computeSequence, EvaluatorRegistry.Default);

            var dataset = new ImageDataset(options.Dataset, graph.Inputs[0].Output.Shape, 0.0f, 1.0f);
            await foreach (var batch in dataset.GetBatchesAsync())
            {
                EvaluateBatch(batch, evaluator);
            }
        }

        private void EvaluateBatch(DenseTensor<float> batch, Evaluator evaluator)
        {
            var input = evaluator.InputAt<float>(0);
            batch.Buffer.Span.CopyTo(input);
            evaluator.Evaluate(dumpDuration: true);

            using (var sw = File.Create("0.bin"))
            {
                sw.Write(evaluator.OutputAt<byte>(0));
            }
        }
    }
}
