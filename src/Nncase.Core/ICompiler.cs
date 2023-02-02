// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase;

/// <summary>
/// Compiler.
/// </summary>
public interface ICompiler
{
    /// <summary>
    /// Import DL model as ir module.
    /// </summary>
    /// <param name="content">Model content.</param>
    /// <returns>Imported ir module.</returns>
    Task<IRModule> ImportModuleAsync(Stream content);

    /// <summary>
    /// import ir module into compiler.
    /// </summary>
    /// <param name="module">Module.</param>
    void ImportIRModule(IRModule module);

    /// <summary>
    /// Compile module.
    /// </summary>
    /// <returns>A <see cref="Task"/> representing the asynchronous operation.</returns>
    Task CompileAsync();

    /// <summary>
    /// Generate code to stream.
    /// </summary>
    /// <param name="output">Stream to be written.</param>
    void Gencode(Stream output);
}
