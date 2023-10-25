// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.XPU;

internal sealed class LinkableModule : ILinkableModule
{
    private const int _textAlignment = 8;

    private readonly Stream _rdata;

    private readonly IReadOnlyList<LinkableFunction> _functions;
    private readonly CompileOptions _options;

    public LinkableModule(Stream rdata, IReadOnlyList<LinkableFunction> functions, CompileOptions options)
    {
        _rdata = rdata;
        _functions = functions;
        _options = options;
    }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        // var csourcePath = LinkCSources();
        if (_options.DumpFlags.HasFlag(DumpFlags.CodeGen))
        {
            foreach (var func in _functions)
            {
                var dumpPath = Path.Join(_options.DumpDir, func.PrimFunction.Name);
                if (!Directory.Exists(dumpPath))
                {
                    Directory.CreateDirectory(dumpPath);
                }

                using (var fs = File.Open(Path.Join(dumpPath, "cluster_def.h"), FileMode.Create))
                {
                    using (var writer = new StreamWriter(fs))
                    {
                        writer.Write(CSourceBuiltn.ClusterDef());
                    }
                }

                using (var fs = File.Open(Path.Join(dumpPath, "main.cpp"), FileMode.Create))
                {
                    using (var writer = new StreamWriter(fs))
                    {
                        writer.Write(func.FunctionCSource.Main);
                    }
                }

                using (var fs = File.Open(Path.Join(dumpPath, "shared.h"), FileMode.Create))
                {
                    using (var writer = new StreamWriter(fs))
                    {
                        writer.Write(func.FunctionCSource.Shared);
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

            // using (var fs = File.Open(Path.Join(dumpPath, "xpuModule.h"), FileMode.Create))
            // {
            //     using (var writer = new StreamWriter(fs))
            //     {
            //         writer.Write(CSourceBuiltn.Header);
            //     }
            // }

            // using (var fs = File.Open(Path.Join(dumpPath, "xpuModule.c"), FileMode.Create))
            // {
            //     File.Open(csourcePath, FileMode.Open, FileAccess.Read).CopyTo(fs);
            // }
        }

        var manager = new SectionManager();
        var textWriter = manager.GetWriter(WellknownSectionNames.Text);
        var linkedFunctions = new List<LinkedFunction>();
        int offset = 0;
        foreach (var func in _functions)
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

    // private string LinkCSources()
    // {
    // writer.WriteLine(CSourceBuiltn.MakeHeader(_functions));
    // foreach (var func in _functions)
    // {
    //     var path = Path.GetTempFileName() + ".cpp";
    //     using (var fs = File.OpenWrite(path))
    //     {
    //         using (var writer = new StreamWriter(fs))
    //         {
    //             writer.WriteLine(CSourceBuiltn.MakeHeader(_functions));
    //         }
    //     }
    // }
    // return path;
    // }
    private string CompileCSource(string sourcePath)
    {
        var compiler = new CSourceCompiler();
        return compiler.Compile(sourcePath, Path.Join(sourcePath, "build", Path.GetFileName(sourcePath)));
    }
}
