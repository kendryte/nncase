using System.Linq;
using System.IO;
using System.Collections.Generic;
using System;
using Nncase.TIR;
using Nncase.IR;

namespace Nncase.Transform
{

    /// <summary>
    /// EGraph pass.
    /// </summary>
    public class TIRPass : FunctionPass
    {
        /// <summary>
        /// Save rules
        /// </summary>
        public readonly List<ExprMutator> Mutators = new();


        /// <summary>
        /// Initializes a new instance of the <see cref="TIRPass"/> class.
        /// </summary>
        /// <param name="name">Name.</param>
        public TIRPass(string name)
            : base(name)
        {
        }

        /// <summary>
        /// add rules
        /// </summary>
        /// <param name="matutors"></param>
        public void Add(params ExprMutator[] matutors)
        {
            Mutators.AddRange(matutors);
        }

        /// <inheritdoc/>
        protected override Function RunCore(Function function, RunPassOptions options)
        {
            options.SetName(Name);
            var pre = function;
            var post = pre;
            RunPassOptions new_options = new(options);
            new_options.SetDir(options.FullDumpDir);
            foreach (var (mutator, i) in Mutators.Select((item, i) => (item, i)))
            {
                new_options.SetName(i + "_" + mutator.GetType().Name);
                OnPassStart(pre, new_options);
                post = (Function)mutator.Visit(pre);
                post.InferenceType();
                OnPassEnd(post, new_options);
                pre = post;
            }
            return post;
        }

        protected override void OnPassStart(Function func, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    ScriptPrinter.DumpAsScript(func, "Start", options.FullDumpDir);
                    break;
                default:
                    break;
            }
        }

        protected override void OnPassEnd(Function func, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    ScriptPrinter.DumpAsScript(func, "End", options.FullDumpDir);
                    break;
                default:
                    break;
            }
        }
    }

}