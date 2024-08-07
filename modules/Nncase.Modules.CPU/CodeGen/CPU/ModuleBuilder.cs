// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// K230CoreModule builder.
/// </summary>
public sealed class CPUModuleBuilder : IModuleBuilder
{
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _rdataWriter;

    public CPUModuleBuilder(CompileOptions options)
    {
        _sectionManager = new();
        _rdataWriter = _sectionManager.GetWriter(WellknownSectionNames.Rdata);
        CompileOptions = options;
    }

    public CompileOptions CompileOptions { get; }

    /// <inheritdoc/>
    public string ModuleKind => "cpu";

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var linkableFunctions = functions.OfType<TIR.PrimFunction>().Select((f, i) => new FunctionBuilder((uint)i, _rdataWriter, (Targets.CpuTargetOptions)CompileOptions.TargetOptions).Build(f)).ToArray();
        _rdataWriter.Flush();

        return new LinkableModule(_sectionManager.GetContent(WellknownSectionNames.Rdata)!, linkableFunctions, CompileOptions);
    }
}
