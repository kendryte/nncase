using Nncase.IR;
using System.Collections.Generic;
using Nncase.Pattern;
using System.IO;
using System;

namespace Nncase.Transform
{

    /// <summary>
    /// dataflow pass
    /// </summary>
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

        /// <inheritdoc/>
        protected override Function RunCore(Function pre, RunPassOptions options)
        {
            if (options.DumpLevel > 0)
            {
                IRPrinter.DumpFunctionAsIL(pre, "Before", Path.Combine(options.FullDumpDir, Name));
            }
            Function post = (Function)DataFlowRewrite.Rewrite(pre, Rules, options);
            if (options.DumpLevel > 0)
            {
                IRPrinter.DumpFunctionAsIL(post, "After", Path.Combine(options.FullDumpDir, Name));
            }
            return post;
        }
    }

    public sealed class ShapeInferPass : FunctionPass
    {
        public readonly List<PatternRule> rules = new()
        {
            new Transform.DataFlow.Rules.FoldConstCall(),
            // new Transform.DataFlow.Rules.FoldConstFunction(),
            new Transform.DataFlow.Rules.FoldShapeOp(),
        };

        public ShapeInferPass(string name = "ShapeInfer") : base(name)
        {
        }

        /// <inheritdoc/>
        protected override Function RunCore(Function pre, RunPassOptions options)
        {
            Function post;
            int count = 0;
            RunPassOptions new_options = new(options);
            new_options.SetDir(options.FullDumpDir);
            while (true)
            {
                post = (Function)DataFlowRewrite.Rewrite(pre, rules, new_options.SetName($"{Name}/Run_{count}"));
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