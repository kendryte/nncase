// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// IR printer.
    /// </summary>
    public class IRPrinter
    {
        /// <summary>
        /// Dump function to IL text.
        /// </summary>
        /// <param name="textWriter">Text writer.</param>
        /// <param name="function">Function.</param>
        /// <returns>Async task.</returns>
        public static Task DumpFunctionAsIL(TextWriter textWriter, Function function)
        {
            var visitor = new ILDumpVisitor(textWriter);
            return visitor.Visit(function);
        }

        private class ILDumpVisitor : ExprFunctor<Task<string>>
        {
            private readonly TextWriter _textWriter;

            public ILDumpVisitor(TextWriter textWriter)
            {
                _textWriter = textWriter;
            }
        }
    }
}
