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
using Nncase.CodeGen.CPU;
using Nncase.Diagnostics;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.CPU;

internal sealed class LinkableModule : ILinkableModule
{
    private readonly Stream _desc;
    private readonly Stream _rdata;
    private readonly IReadOnlyList<Stream> _localRdatas;

    private readonly IReadOnlyList<ILinkableFunction> _functions;

    public LinkableModule(Stream desc, Stream rdata, IReadOnlyList<Stream> localRdatas, IReadOnlyList<ILinkableFunction> functions, CompileOptions options)
    {
        _desc = desc;
        _rdata = rdata;
        _localRdatas = localRdatas;
        _functions = functions;
    }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        var codegenDir = DumpScope.Current.Directory;
        {
            if (!Directory.Exists(codegenDir))
            {
                Directory.CreateDirectory(codegenDir);
            }

            using (var writer = new StreamWriter(File.Open(Path.Join(codegenDir, "device.h"), FileMode.Create)))
            {
                writer.Write(CSourceBuiltn.KernelHeader);

                foreach (var func in _functions.OfType<LinkableDeviceFunction>())
                {
                    writer.Write(func.Header);
                }
            }
        }

        foreach (var func in _functions.OfType<LinkableKernelFunction>())
        {
            var dumpPath = Path.Join(codegenDir, func.PrimFunction.Name);
            if (!Directory.Exists(dumpPath))
            {
                Directory.CreateDirectory(dumpPath);
            }

            using (var fs = File.Open(Path.Join(dumpPath, "thread_main.cpp"), FileMode.Create))
            {
                using (var writer = new StreamWriter(fs))
                {
                    writer.Write(func.FunctionCSource.Main);
                }
            }

            using (var fs = File.Open(Path.Join(dumpPath, "kernel.h"), FileMode.Create))
            {
                using (var writer = new StreamWriter(fs))
                {
                    writer.Write(func.FunctionCSource.Kernel);
                }
            }

            using (var fs = File.Open(Path.Join(dumpPath, "topo_aware_runtime.h"), FileMode.Create))
            {
                using (var writer = new StreamWriter(fs))
                {
                    writer.Write(func.FunctionCSource.TopoRuntime);
                }
            }

            using (var fs = File.Open(Path.Join(dumpPath, "topology_def.h"), FileMode.Create))
            {
                using (var writer = new StreamWriter(fs))
                {
                    writer.Write(func.FunctionCSource.TopologyDef);
                }
            }

            using (var fs = File.Open(Path.Join(dumpPath, "CMakeLists.txt"), FileMode.Create))
            {
                using (var writer = new StreamWriter(fs))
                {
                    writer.Write(CSourceBuiltn.CMakeDef(func.PrimFunction.Name));
                }
            }
        }

        var manager = new SectionManager();
        var textWriter = manager.GetWriter(WellknownSectionNames.Text);
        var linkedFunctions = new List<LinkedFunction>();
        int offset = 0;
        ulong rdataAlign = 8;
        foreach (var func in _functions.OfType<LinkableKernelFunction>())
        {
            rdataAlign = Math.Max(rdataAlign, func.PrimFunction.SchedResult.DataAlign);
            var dumpPath = Path.Join(codegenDir, func.PrimFunction.Name);
            var elfPath = CompileCSource(dumpPath);

            var func_text = File.ReadAllBytes(elfPath);
            textWriter.Write(func_text);
            linkedFunctions.Add(new LinkedFunction(func.Id, func.SourceFunction, (uint)offset, (uint)func_text.Length, func.Sections));
            offset += func_text.Length;
        }

        return new LinkedModule(linkedFunctions, _desc, manager.GetContent(WellknownSectionNames.Text)!, _rdata, _localRdatas, rdataAlign);
    }

    private string CompileCSource(string sourcePath)
    {
        var compiler = new CSourceCompiler();
        var binDir = Path.Join(sourcePath, "build", "nncase_ntt_module");
        return compiler.Compile(sourcePath, binDir);
    }
}
