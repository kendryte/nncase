using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NnCase.IR;
using NnCase.IR.Operators;

namespace NnCase.Transforms
{
    public class FoldQuantizeTransform : Transform
    {
        protected override bool SkipSelfContainedCheck => true;

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is Quantize q)
            {
                if (NodeTreeHelper.TryGetDirectChild<Dequantize>(q, out var deq)
                    && q.QuantizationParam.CloseTo(deq.QuantizationParam))
                {
                    context.Inputs.Add(q.Input);
                    context.Outputs.Add(deq.Output);

                    context.MatchedNodes.Add(q);
                    context.MatchedNodes.Add(deq);
                    return true;
                }
            }
            else if (node is Dequantize deq)
            {
                if (NodeTreeHelper.TryGetDirectChild<Quantize>(deq, out q)
                    && q.QuantizationParam.CloseTo(deq.QuantizationParam))
                {
                    context.Inputs.Add(deq.Input);
                    context.Outputs.Add(q.Output);

                    context.MatchedNodes.Add(deq);
                    context.MatchedNodes.Add(q);
                    return true;
                }
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;

            foreach (var input in inputs.ToList())
                input.Connect(output);
        }
    }
}
