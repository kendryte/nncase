// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// K230CoreModule builder.
/// </summary>
public sealed class ModuleBuilder : IModuleBuilder, IDisposable
{
    private readonly MemoryStream _rdataContent = new MemoryStream();
    private readonly BinaryWriter _rdataWriter;

    public ModuleBuilder(CompileOptions options)
    {
        _rdataWriter = new BinaryWriter(_rdataContent, Encoding.UTF8, leaveOpen: true);
        CompileOptions = options;
    }

    public CompileOptions CompileOptions { get; }

    /// <inheritdoc/>
    public string ModuleKind => Targets.CPUTarget.Kind;

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var linkableFunctions = functions.OfType<TIR.PrimFunction>().Select((f, i) => new FunctionBuilder((uint)i, _rdataWriter).Build(f)).ToArray();
        _rdataWriter.Flush();

        return new LinkableModule(_rdataContent.ToArray(), linkableFunctions, CompileOptions);
    }

    public void Dispose() => ((IDisposable)_rdataContent).Dispose();
}
