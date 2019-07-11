using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class ConstantFoldingTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node.Inputs.Any() && node.Inputs.All(x => x.Connection.Owner is Constant) &&
                node.Outputs.All(x => x.MemoryType == MemoryType.Constant || x.MemoryType == MemoryType.Main))
            {
                context.Outputs.AddRange(node.Outputs);
                context.MatchedNodes.Add(node);
                context.MatchedNodes.AddRange(node.Inputs.Select(x => x.Connection.Owner));
                return true;
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var inputs = context.Outputs[0].Connections.ToList();

            var newGraph = new Graph();
            foreach (var output in context.Outputs)
            {
                var node = newGraph.AddNode(new OutputNode(output.Type, output.Shape));
                node.Input.Connect(output);
            }

            var allocators = new Dictionary<MemoryType, MemoryAllocator>
            {
                { MemoryType.Constant, new MemoryAllocator() },
                { MemoryType.Main, new MemoryAllocator() }
            };
            var allocationContext = new AllocationContext(allocators);
            var computeSequence = new List<Node>();
            Scheduler.Schedule(newGraph.Outputs, allocationContext, computeSequence);

            var evaluator = new Evaluator(allocators, allocationContext.Allocations, computeSequence, EvaluatorRegistry.Default);
            evaluator.Evaluate();

            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var data = evaluator.OutputAt<byte>(i);
                var constant = context.Graph.AddNode(new Constant(input.Type, data.ToArray(), input.Shape));
                input.Connect(constant.Output);
            }

            foreach (var output in newGraph.Outputs)
                output.Input.ClearConnection();
        }
    }
}
