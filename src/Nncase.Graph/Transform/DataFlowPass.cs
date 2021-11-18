using Nncase.IR;
using System.Collections.Generic;
using Nncase.Pattern;
using System.IO;

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

        protected override void RunCore(Function function, RunPassOptions options)
        {
            if (options.DumpIR)
            {
                IRPrinter.DumpFunctionAsIL(Path.Combine(options.DumpDir, Name), function, "Before");
            }
            DataFlowRewrite.Rewrite(function, Rules);
            if (options.DumpIR)
            {
                IRPrinter.DumpFunctionAsIL(Path.Combine(options.DumpDir, Name), function, "After");
            }
        }
    }

    public sealed class ShapeInferPass : FunctionPass
    {
        public readonly List<PatternRule> rules = new()
        {
            new Transform.DataFlow.Rules.FoldConstCall(),
            new Transform.DataFlow.Rules.FoldShapeOp(),
        };

        public ShapeInferPass(string name) : base(name)
        {
        }

        protected override void RunCore(Function function, RunPassOptions options)
        {

            if (options.DumpIR)
            {
                IRPrinter.DumpFunctionAsIL(Path.Combine(options.DumpDir, Name), function, "Before");
            }

            while (!TypeInference.InferenceType(function))
            {
                DataFlowRewrite.Rewrite(function, rules);
            }
            if (options.DumpIR)
            {
                IRPrinter.DumpFunctionAsIL(Path.Combine(options.DumpDir, Name), function, "After");
            }
        }

    }
}