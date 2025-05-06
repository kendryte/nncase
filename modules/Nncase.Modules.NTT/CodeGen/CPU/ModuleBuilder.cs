// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Targets;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// K230CoreModule builder.
/// </summary>
public sealed class NTTModuleBuilder : IModuleBuilder
{
    private readonly SectionManager _sectionManager;
    private readonly BinaryWriter _rdataWriter;
    private readonly BinaryWriter[] _localRdataWriters;

    public NTTModuleBuilder(CompileOptions options)
    {
        _sectionManager = new();
        _rdataWriter = _sectionManager.GetWriter(WellknownSectionNames.Rdata);
        var shardCount = TensorUtilities.GetProduct(((Targets.CpuTargetOptions)options.TargetOptions).Hierarchies[0]);
        _localRdataWriters = new BinaryWriter[shardCount];
        for (int i = 0; i < shardCount; i++)
        {
            _localRdataWriters[i] = _sectionManager.GetWriter(WellknownSectionNames.LocalRdata, i);
        }

        CompileOptions = options;
    }

    public CompileOptions CompileOptions { get; }

    /// <inheritdoc/>
    public string ModuleKind => "cpu";

    /// <inheritdoc/>
    public ILinkableModule Build(IReadOnlyList<BaseFunction> functions)
    {
        var targetOptions = (CpuTargetOptions)CompileOptions.TargetOptions;

        // 1. write the module header
        using (var writer = _sectionManager.GetWriter(LinkedModule.KernelHeaderSectionName))
        {
            var header = default(DescHeader);
            header.ThreadDim = (uint)targetOptions.Hierarchies[0][^1];
            header.BlockDim = targetOptions.Hierarchies[0].Length < 2 ? 1 : (uint)targetOptions.Hierarchies[0][^2];
            header.ChipDim = targetOptions.Hierarchies[0].Length < 3 ? 1 : (uint)targetOptions.Hierarchies[0][^3];
            writer.Write(ref header);
        }

        var linkableFunctions = functions.OfType<TIR.PrimFunction>().Select((f, i) => new FunctionBuilder((uint)i, _rdataWriter, _localRdataWriters, (Targets.CpuTargetOptions)CompileOptions.TargetOptions).Build(f)).ToArray();
        _rdataWriter.Flush();
        var localRdataContents = Enumerable.Range(0, _localRdataWriters.Length).Select(i =>
        {
            _localRdataWriters[i].Flush();
            return _sectionManager.GetContent(WellknownSectionNames.LocalRdata, i)!;
        }).ToArray();

        return new LinkableModule(_sectionManager.GetContent(LinkedModule.KernelHeaderSectionName)!, _sectionManager.GetContent(WellknownSectionNames.Rdata)!, localRdataContents, linkableFunctions, CompileOptions);
    }
}
