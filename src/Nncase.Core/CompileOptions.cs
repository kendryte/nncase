// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;

namespace Nncase.Compiler
{
    internal class CompileOptions { }
    
    /// <summary>
    /// Options of compile command.
    /// </summary>
    public interface ICompileOptions
    {
        /// <summary>
        /// Gets or sets input file.
        /// </summary>
        public string InputFile { get; set; }

        /// <summary>
        /// Gets or sets the import model format.
        /// </summary>
        public string InputFormat { get; set; }

        /// <summary>
        /// Gets or sets target.
        /// </summary>
        public string Target { get; set; }

        /// <summary>
        /// Gets or sets the dump level.
        /// </summary>
        public int DumpLevel { get; set; }

        /// <summary>
        /// Gets or sets the dump directory.
        /// </summary>
        public string DumpDir { get; set; }

        public bool UsePTQ { get; set; }
    }
}
