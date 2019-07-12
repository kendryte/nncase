using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics.Tensors;
using System.Text;
using System.Threading.Tasks;
using NnCase.CodeGen;
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
            var quantizer = new Quantizer();
            var target = CreateTarget(options);
            target.RegisterEvaluators(EvaluatorRegistry.Default);
            target.RegisterEmitters(CodeGenRegistry.Default);

            // 2. Optimize Pass 1 (Generic)
            OptimizePass1(graph);

            // 3. Optimize Pass 2 (Target aware)
            OptimizePass2(graph, target);

            // 4. Quantize
            // await Quantize(options, graph, target, quantizer);

            // 5. Simulate
            await Simulate(options, graph, target);

            // 6. CodeGen
            GenerateCode(options, graph, target);
        }

        private Target CreateTarget(Options options)
        {
            switch (options.Target)
            {
                case "cpu":
                    return new Targets.CPU.CPUTarget();
                case "k210":
                    return new Targets.K210.K210Target();
                default:
                    throw new NotSupportedException($"Unsupported target: {options.Target}");
            }
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

        private async Task Quantize(Options options, Graph graph, Target target, Quantizer quantizer)
        {
            // 3.1. Add quantization checkpoints
            AddQuantizationCheckpoints(graph, target);

            // 3.2 Get activation ranges
            await GetActivationRanges(options, graph, target, quantizer);

            // 3.3 Quantize graph
            QuantizeGraph(graph, target, quantizer);
        }

        private void QuantizeGraph(Graph graph, Target target, Quantizer quantizer)
        {
            target.QuantizeGraph(graph, quantizer);
            DumpGraph(graph, "optimize_3");
        }

        private void AddQuantizationCheckpoints(Graph graph, Target target)
        {
            target.AddQuantizationCheckpoints(graph);
        }

        private async Task GetActivationRanges(Options options, Graph graph, Target target, Quantizer quantizer)
        {
            var allocators = new Dictionary<MemoryType, MemoryAllocator>();
            target.AddAllocators(allocators);
            var allocationContext = new AllocationContext(allocators);
            var computeSequence = new List<Node>();
            Scheduler.Schedule(graph.Outputs, allocationContext, computeSequence);

            var evaluator = new Evaluator(allocators, allocationContext.Allocations, computeSequence, EvaluatorRegistry.Default);

            var dataset = new ImageDataset(options.Dataset, graph.Inputs[0].Output.Shape, 0.0f, 1.0f);
            await foreach (var batch in dataset.GetBatchesAsync())
            {
                EvaluateBatch(batch, evaluator, quantizer);
            }
        }

        private void EvaluateBatch(DenseTensor<float> batch, Evaluator evaluator, Quantizer quantizer, bool dumpDuration = false, bool dumpOutput = false)
        {
            var input = evaluator.InputAt<float>(0);
            batch.Buffer.Span.CopyTo(input);
            evaluator.Evaluate(quantizer, dumpDuration: dumpDuration);

            if (dumpOutput)
            {
                using (var sw = File.Create("0.bin"))
                {
                    sw.Write(evaluator.OutputAt<byte>(0));
                }
            }
        }

        private async Task Simulate(Options options, Graph graph, Target target)
        {
            var allocators = new Dictionary<MemoryType, MemoryAllocator>();
            target.AddAllocators(allocators);
            var allocationContext = new AllocationContext(allocators);
            var computeSequence = new List<Node>();
            Scheduler.Schedule(graph.Outputs, allocationContext, computeSequence);

            var evaluator = new Evaluator(allocators, allocationContext.Allocations, computeSequence, EvaluatorRegistry.Default);

            var dataset = new ImageDataset(options.Dataset, graph.Inputs[0].Output.Shape, 0.0f, 1.0f);
            await foreach (var batch in dataset.GetBatchesAsync())
            {
                EvaluateBatch(batch, evaluator, null, dumpOutput: true, dumpDuration: true);
            }
        }

        private void GenerateCode(Options options, Graph graph, Target target)
        {
            var allocators = new Dictionary<MemoryType, MemoryAllocator>();
            target.AddAllocators(allocators);
            var allocationContext = new AllocationContext(allocators);
            var computeSequence = new List<Node>();
            Scheduler.Schedule(graph.Outputs, allocationContext, computeSequence);

            var generator = new Generator(allocators, allocationContext.Allocations, computeSequence, CodeGenRegistry.Default);
            generator.Generate(File.Create(options.Output));
        }
    }
}
