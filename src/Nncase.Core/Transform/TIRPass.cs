// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
    /// TIR Mutator Pass.
    /// Because of we will mutate the expression multiple times, so use MutatorCreator create the new mutator.
    /// </summary>
    public class TIRPass : FunctionPass, IEnumerable<Func<ExprMutator>>
    {
        /// <summary>
        /// Save rules.
        /// </summary>
        public readonly List<Func<ExprMutator>> MutatorCreators = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="TIRPass"/> class.
        /// </summary>
        /// <param name="name">Name.</param>
        public TIRPass(string name)
            : base(name)
        {
        }

        /// <inheritdoc/>
        protected override Callable RunCore(Callable callable, RunPassOptions options)
        {
            var pass_options = options.SetPassName(Name);
            var post = callable;
            var last = post;
            int count = 0;
            var typeinfer_ret = true;
            OnPassStart(last, pass_options);
            do
            {
                bool isMutated = false;
                foreach (var creator in MutatorCreators)
                {
                    var mutator = creator();
                    last = post;
                    post = (Callable)mutator.Visit(last);
                    if (mutator.IsMutated)
                    {
                        isMutated = true;
                        typeinfer_ret = CompilerServices.InferenceType(post);
                        OnMutated(post, $"{count++}_{mutator.GetType().Name}", pass_options);
                        if (!typeinfer_ret) throw new InvalidOperationException($"After Run Mutator {count - 1}_{mutator.GetType().Name} , The Type Inference Failed!");
                        break;
                    }
                }

                if (!isMutated)
                    break;
            } while (true);

            OnPassEnd(post, pass_options);
            return post;
        }

        void OnMutated(Callable callable, string prefix, RunPassOptions options)
        {
            switch (options.DumpLevel)
            {
                case >= 2:
                    CompilerServices.DumpIR(callable, prefix, options.PassDumpDir);
                    break;
                case >= 1:
                    break;
                default:
                    break;
            }
        }

        /// <inheritdoc/>
        public IEnumerator<Func<ExprMutator>> GetEnumerator()
        {
            return ((IEnumerable<Func<ExprMutator>>)MutatorCreators).GetEnumerator();

        }

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)MutatorCreators).GetEnumerator();
        }

        /// <summary>
        /// add the mutator
        /// </summary>
        /// <param name="mutator"></param>
        public void Add(Func<ExprMutator> mutator)
        {
            MutatorCreators.Add(mutator);
        }
    }
}