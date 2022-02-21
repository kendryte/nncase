// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform
{
    /// <summary>
    /// EGraph pass.
    /// </summary>
    public class EGraphPass : FunctionPass
    {
        /// <summary>
        /// Save rules.
        /// </summary>
        public readonly List<IRewriteRule> Rules = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="EGraphPass"/> class.
        /// </summary>
        /// <param name="name">Name.</param>
        public EGraphPass(string name)
            : base(name)
        {
        }

        /// <summary>
        /// add rules.
        /// </summary>
        /// <param name="rules"></param>
        public void Add(params IRewriteRule[] rules)
        {
            foreach (var rule in rules)
            {
                Rules.Add(rule);
            }
        }

        /// <inheritdoc/>
        protected override Function RunCore(Function function, RunPassOptions options)
        {
            options.SetName(Name);
            var graph = new EGraph();
            graph.Add(function);
            EGraphReWriter.ReWrite(graph, Rules, options);
            return function;
        }
    }
}
