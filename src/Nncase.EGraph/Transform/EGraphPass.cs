// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform
{
    /// <summary>
    /// EGraph pass.
    /// </summary>
    public class EGraphPass : FunctionPass
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EGraphPass"/> class.
        /// </summary>
        /// <param name="name">Name.</param>
        public EGraphPass(string name)
            : base(name)
        {
        }

        /// <inheritdoc/>
        protected override Function RunCore(Function function, RunPassOptions options)
        {
            // var graph = new EGraph();
            // graph.Add(function);
            return function;
        }
    }
}
