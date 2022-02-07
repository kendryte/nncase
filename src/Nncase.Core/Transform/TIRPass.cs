using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Transform
{

    /// <summary>
    /// EGraph pass.
    /// </summary>
    public class TIRPass : FunctionPass, IEnumerable<ExprMutator>
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

        /// <inheritdoc/>
        protected override Function RunCore(Function function, RunPassOptions options)
        {
            options.SetName(Name);
            var pre = function;
            var post = pre;
            RunPassOptions new_options = new(options);
            new_options.SetDir(options.PassDumpDir);
            foreach (var (mutator, i) in Mutators.Select((item, i) => (item, i)))
            {
                new_options.SetName(i + "_" + mutator.GetType().Name);
                OnPassStart(pre, new_options);
                post = (Function)mutator.Visit(pre);
                var inferRes = post.InferenceType();
                OnPassEnd(post, new_options);
                if (!inferRes) throw new InvalidOperationException("After Run Pass, The Type Inference Failed!");
                pre = post;
            }
            return post;
        }

        /// <inheritdoc/>
        protected override void OnPassStart(Function func, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    ScriptPrinter.DumpAsScript(func, "Start", options.PassDumpDir);
                    break;
                default:
                    break;
            }
        }

        /// <inheritdoc/>
        protected override void OnPassEnd(Function func, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    ScriptPrinter.DumpAsScript(func, "End", options.PassDumpDir);
                    break;
                default:
                    break;
            }
        }

        /// <inheritdoc/>
        public IEnumerator<ExprMutator> GetEnumerator()
        {
            return ((IEnumerable<ExprMutator>)Mutators).GetEnumerator();

        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)Mutators).GetEnumerator();
        }

        /// <summary>
        /// add the mutator
        /// </summary>
        /// <param name="mutator"></param>
        public void Add(ExprMutator mutator)
        {
            Mutators.Add(mutator);
        }
    }

}