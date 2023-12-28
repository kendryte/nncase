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
    private readonly Stream _rdata;

    private readonly IReadOnlyList<ILinkableFunction> _functions;
    private readonly CompileOptions _options;

    public LinkableModule(Stream rdata, IReadOnlyList<ILinkableFunction> functions, CompileOptions options)
    {
        _rdata = rdata;
        _functions = functions;
        _options = options;
    }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        {
            if (!Directory.Exists(_options.DumpDir))
            {
                Directory.CreateDirectory(_options.DumpDir);
            }

            using (var writer = new StreamWriter(File.Open(Path.Join(_options.DumpDir, "device.h"), FileMode.Create)))
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
            var dumpPath = Path.Join(_options.DumpDir, func.PrimFunction.Name);
            if (!Directory.Exists(dumpPath))
            {
                Directory.CreateDirectory(dumpPath);
            }

            using (var fs = File.Open(Path.Join(dumpPath, "main.cpp"), FileMode.Create))
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
        foreach (var func in _functions.OfType<LinkableKernelFunction>())
        {
            var dumpPath = Path.Join(_options.DumpDir, func.PrimFunction.Name);
            var elfPath = CompileCSource(dumpPath);

            var func_text = File.ReadAllBytes(elfPath);
            textWriter.Write(func_text);
            linkedFunctions.Add(new LinkedFunction(func.Id, func.SourceFunction, (uint)offset, (uint)func_text.Length, func.Sections));
            offset += func_text.Length;
        }

        return new LinkedModule(linkedFunctions, manager.GetContent(WellknownSectionNames.Text)!, _rdata);
    }

    private string CompileCSource(string sourcePath)
    {
        var compiler = new CSourceCompiler();
        var binDir = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? Path.Join(sourcePath, "build", "nncase_cpu_module.exe")
            : Path.Join(sourcePath, "build", "nncase_cpu_module");
        return compiler.Compile(sourcePath, binDir);
    }
}
