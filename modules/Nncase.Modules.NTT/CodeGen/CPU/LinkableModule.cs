// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.CodeGen.NTT;
using Nncase.Diagnostics;
using Nncase.Targets;

namespace Nncase.CodeGen.NTT;

internal sealed class LinkableModule : ILinkableModule
{
    private readonly Stream _desc;
    private readonly Stream _rdata;
    private readonly IReadOnlyList<Stream> _threadLocalRdatas;
    private readonly IReadOnlyList<Stream> _blockLocalRdatas;
    private readonly IReadOnlyList<ILinkableFunction> _functions;
    private readonly NTTTargetOptions _targetOptions;

    public LinkableModule(Stream desc, Stream rdata, IReadOnlyList<Stream> threadLocalRdatas, IReadOnlyList<Stream> blockLocalRdatas, IReadOnlyList<ILinkableFunction> functions, CompileOptions options)
    {
        _desc = desc;
        _rdata = rdata;
        _threadLocalRdatas = threadLocalRdatas;
        _blockLocalRdatas = blockLocalRdatas;
        _functions = functions;
        PublicFunctions = _functions.OfType<LinkableKernelFunction>().ToArray();
        _targetOptions = (NTTTargetOptions)options.TargetOptions;
    }

    public IReadOnlyList<ILinkableFunction> PublicFunctions { get; }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        var moduleKind = _functions[0].SourceFunction.ModuleKind;
        var codegenDir = Path.Join(DumpScope.Current.Directory, "CodeGen", moduleKind);
        if (!Directory.Exists(codegenDir))
        {
            Directory.CreateDirectory(codegenDir);
        }

        WriteDeviceFunctions(codegenDir);
        var kernelFiles = WriteKernelFunctions(codegenDir);
        WriteTopoAwareRuntime(codegenDir);
        WriteModuleTopologyDef(codegenDir);

        var mainFunc = (LinkableKernelFunction)PublicFunctions.First(x => x.SourceFunction.IsEntry);
        WriteThreadMain(codegenDir, mainFunc, kernelFiles);
        WriteCMakeLists(codegenDir);

        return GenerateLinkedModule(codegenDir, mainFunc);
    }

    private void WriteDeviceFunctions(string codegenDir)
    {
        using (var writer = new StreamWriter(File.Open(Path.Join(codegenDir, "device_functions.h"), FileMode.Create)))
        {
            writer.Write(CSourceBuiltn.DeviceHeader);

            foreach (var func in _functions.OfType<LinkableDeviceFunction>())
            {
                writer.Write(func.Header);
            }
        }
    }

    private IReadOnlyList<string> WriteKernelFunctions(string codegenDir)
    {
        using (var writer = new StreamWriter(File.Open(Path.Join(codegenDir, "kernel_functions.h"), FileMode.Create)))
        {
            writer.Write(CSourceBuiltn.KernelDeclareHeader);

            foreach (var func in _functions.OfType<LinkableKernelFunction>())
            {
                writer.Write(func.FunctionCSource.Declare);
            }
        }

        var kernelFiles = new List<string>();
        foreach (var func in _functions.OfType<LinkableKernelFunction>())
        {
            var fileName = $"{func.PrimFunction.Name}.h";
            kernelFiles.Add(fileName);
            using (var writer = new StreamWriter(File.Open(Path.Join(codegenDir, fileName), FileMode.Create)))
            {
                writer.Write(func.FunctionCSource.Kernel);
            }
        }

        return kernelFiles;
    }

    private void WriteTopoAwareRuntime(string codegenDir)
    {
        var scheduleResults = _functions.OfType<LinkableKernelFunction>();
        var alignment = scheduleResults.Max(x => x.PrimFunction.SchedResult.DataAlign);

        // FIXME: This is a temporary fix, we should use the bufferize pass to get the collective pool size.
        var collectivePoolSize = scheduleResults.Max(x => x.FunctionCSource.CollectivePoolSize);
        using (var fs = File.Open(Path.Join(codegenDir, "topo_aware_runtime.h"), FileMode.Create))
        {
            using (var writer = new StreamWriter(fs))
            {
                writer.Write(CSourceBuiltn.TopoAwareRuntimeDef(_targetOptions, alignment, collectivePoolSize));
            }
        }
    }

    private void WriteModuleTopologyDef(string codegenDir)
    {
        using (var fs = File.Open(Path.Join(codegenDir, "module_topology_def.h"), FileMode.Create))
        {
            using (var writer = new StreamWriter(fs))
            {
                writer.Write(CSourceBuiltn.ModuleTopologyDef(_targetOptions));
            }
        }
    }

    private void WriteThreadMain(string codegenDir, LinkableKernelFunction mainFunc, IReadOnlyList<string> kernelFiles)
    {
        using (var fs = File.Open(Path.Join(codegenDir, "thread_main.cpp"), FileMode.Create))
        {
            using (var writer = new StreamWriter(fs))
            {
                var scheduleResult = mainFunc.SourceFunction.SchedResult;
                var memoryPoolDesc = mainFunc.MemoryPoolDesc;

                writer.Write(CSourceBuiltn.ThreadMainHeader);
                foreach (var file in kernelFiles)
                {
                    writer.WriteLine($"#include \"{file}\"");
                }

                writer.Write(CSourceBuiltn.MakeMain(
                    primFunction: (TIR.PrimFunction)mainFunc.SourceFunction,
                    dataAlign: scheduleResult.DataAlign,
                    dataUsage: scheduleResult.DataUsage,
                    rdataPoolSize: memoryPoolDesc.RdataPoolSize,
                    threadLocalRdataPoolSize: memoryPoolDesc.ThreadLocalRdataPoolSize,
                    blockLocalRdataPoolSize: memoryPoolDesc.BlockLocalRdataPoolSize,
                    options: _targetOptions));
            }
        }
    }

    private void WriteCMakeLists(string codegenDir)
    {
        using (var fs = File.Open(Path.Join(codegenDir, "CMakeLists.txt"), FileMode.Create))
        {
            using (var writer = new StreamWriter(fs))
            {
                writer.Write(CSourceBuiltn.CMakeDef());
            }
        }
    }

    private ILinkedModule GenerateLinkedModule(string codegenDir, LinkableKernelFunction mainFunc)
    {
        var manager = new SectionManager();
        var textWriter = manager.GetWriter(WellknownSectionNames.Text);
        var linkedFunctions = new List<LinkedFunction>();
        ulong rdataAlign = 8;

        foreach (var func in _functions.OfType<LinkableKernelFunction>())
        {
            rdataAlign = Math.Max(rdataAlign, func.PrimFunction.SchedResult.DataAlign);
        }

        var elfPath = CompileCSource(codegenDir);
        var funcText = File.ReadAllBytes(elfPath);
        textWriter.Write(funcText);
        linkedFunctions.Add(new LinkedFunction(mainFunc.Id, mainFunc.SourceFunction, 0, (uint)funcText.Length, mainFunc.Sections));
        return new LinkedModule(linkedFunctions, _desc, manager.GetContent(WellknownSectionNames.Text)!, _rdata, _threadLocalRdatas, _blockLocalRdatas, rdataAlign);
    }

    private string CompileCSource(string sourcePath)
    {
        var compiler = new CSourceCompiler();
        var binDir = Path.Join(sourcePath, "build", "nncase_ntt_module");
        return compiler.Compile(sourcePath, binDir);
    }
}
