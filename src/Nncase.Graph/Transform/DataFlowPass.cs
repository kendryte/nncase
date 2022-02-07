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
    public class DataFlowPass : FunctionPass
    {
        public readonly List<PatternRule> Rules = new();

        public DataFlowPass(string name) : base(name)
        {
        }

        /// <summary>
        /// add the pattern rules
        /// </summary>
        /// <param name="rules"></param>
        public void Add(params PatternRule[] rules) => Rules.AddRange(rules);

        /// <summary>
        /// <see cref="Add(PatternRule[])"/>
        /// </summary>
        /// <param name="rules"></param>
        public void Add(IEnumerable<PatternRule> rules) => Rules.AddRange(rules);

        /// <summary>
        /// the callback function you can custom process func with run pass options
        /// </summary>
        /// <param name="func"> func without run pass</param>
        /// <param name="options"></param>
        protected override void OnPassStart(Function func, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    IRPrinter.DumpFunctionAsIL(func, "Start", Path.Combine(options.PassDumpDir, Name));
                    break;
                case >= 1:
                    Console.WriteLine($"On {Name} Pass Start:");
                    func.DumpExprAsIL();
                    break;
                default:
                    break;
            }
        }

        /// <summary>
        /// the callback function you can custom process func with run pass options
        /// </summary>
        /// <param name="func"> func with rewrited. </param>
        /// <param name="options"></param>
        protected override void OnPassEnd(Function func, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    IRPrinter.DumpFunctionAsIL(func, "End", Path.Combine(options.PassDumpDir, Name));
                    break;
                case >= 1:
                    Console.WriteLine($"On {Name} Pass End:");
                    func.DumpExprAsIL();
                    break;
                default:
                    break;
            }
        }

        /// <inheritdoc/>
        protected override Function RunCore(Function pre, RunPassOptions options)
        {
            OnPassStart(pre, options);
            Function post = (Function)DataFlowRewrite.Rewrite(pre, Rules, options);
            OnPassEnd(post, options);
            return post;
        }
    }

    public sealed class ShapeInferPass : DataFlowPass
    {

        public ShapeInferPass(string name = "ShapeInfer") : base(name)
        {
            Rules.Add(new Transform.Rule.FoldConstCall());
            Rules.Add(new Transform.Rule.FoldShapeOp());
        }

        /// <inheritdoc/>
        protected override Function RunCore(Function pre, RunPassOptions options)
        {
            Function post;
            int count = 0;
            RunPassOptions new_options = new(options);
            new_options.SetDir(options.PassDumpDir);
            while (true)
            {
                post = (Function)DataFlowRewrite.Rewrite(pre, Rules, new_options.SetName($"{Name}/Run_{count}"));
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