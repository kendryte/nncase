using Nncase.IR;
using System.Collections.Generic;
using Nncase.Pattern;
using System.IO;
using System;

namespace Nncase.Transform
{
    public sealed class DataFlowPass : FunctionPass
    {
        public readonly List<PatternRule> Rules = new();

        public DataFlowPass(string name) : base(name)
        {
        }

        public void Add(params PatternRule[] rules)
        {
            foreach (var rule in rules)
            {
                Rules.Add(rule);
            }
        }

        protected override Function RunCore(Function pre, RunPassOptions options)
        {
            if (options.DumpIR)
            {
                IRPrinter.DumpFunctionAsIL(Path.Combine(options.DumpDir, Name), pre, "Before");
            }
            Function post = (Function)DataFlowRewrite.Rewrite(pre, Rules);
            if (options.DumpIR)
            {
                IRPrinter.DumpFunctionAsIL(Path.Combine(options.DumpDir, Name), post, "After");
            }
            return post;
        }
    }

    public sealed class ShapeInferPass : FunctionPass
    {
        public readonly List<PatternRule> rules = new()
        {
            new Transform.DataFlow.Rules.FoldConstCall(),
            new Transform.DataFlow.Rules.FoldConstFunction(),
            new Transform.DataFlow.Rules.FoldShapeOp(),
        };

        public ShapeInferPass() : base("ShapeInfer")
        {
        }

        protected override Function RunCore(Function pre, RunPassOptions options)
        {
            Function post;
            int count = 0;
            var dumpPath = Path.Combine(options.DumpDir, Name);
            while (true)
            {
                if (options.DumpIR)
                    IRPrinter.DumpFunctionAsIL(dumpPath, pre, $"{count}_Before");

                post = (Function)DataFlowRewrite.Rewrite(pre, rules);

                if (options.DumpIR)
                    IRPrinter.DumpFunctionAsIL(dumpPath, post, $"{count++}_After");

                if (post == pre)
                {
                    if (!TypeInference.InferenceType(post))
                        throw new InvalidOperationException("Can't InferShape For This Model!");
                    else
                        break;
                }
                pre = post;
            }
            return post;
        }

    }
}