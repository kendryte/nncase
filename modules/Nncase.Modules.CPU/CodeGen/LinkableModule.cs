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

namespace Nncase.CodeGen.CPU;

internal sealed class LinkableModule : ILinkableModule
{
    private const int _textAlignment = 8;

    private readonly byte[] _rdata;

    private readonly IReadOnlyList<LinkableFunction> _functions;
    private readonly CompileOptions _options;

    public LinkableModule(byte[] rdata, IReadOnlyList<LinkableFunction> functions, CompileOptions options)
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
                        writer.Write(CSourceBuiltn.MakeMain(func.PrimFunction));
                    }
                }

                using (var fs = File.Open(Path.Join(dumpPath, "shared.h"), FileMode.Create))
                {
                    using (var writer = new StreamWriter(fs))
                    {
                        writer.Write(CSourceBuiltn.MakeShared());
                    }
                }

                using (var fs = File.Open(Path.Join(dumpPath, "kernel.h"), FileMode.Create))
                {
                    using (var writer = new StreamWriter(fs))
                    {
                        writer.Write(CSourceBuiltn.KernelHeader);
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

            // using (var fs = File.Open(Path.Join(dumpPath, "cpuModule.h"), FileMode.Create))
            // {
            //     using (var writer = new StreamWriter(fs))
            //     {
            //         writer.Write(CSourceBuiltn.Header);
            //     }
            // }

            // using (var fs = File.Open(Path.Join(dumpPath, "cpuModule.c"), FileMode.Create))
            // {
            //     File.Open(csourcePath, FileMode.Open, FileAccess.Read).CopyTo(fs);
            // }
        }

        var text = new List<byte>();
        var linkedFunctions = new List<LinkedFunction>();
        int offset = 0;
        foreach (var func in _functions)
        {
            var dumpPath = Path.Join(_options.DumpDir, func.PrimFunction.Name);
            var elfPath = CompileCSource(dumpPath);

            var func_text = File.ReadAllBytes(elfPath);
            text.AddRange(func_text);
            linkedFunctions.Add(new LinkedFunction(func.Id, func.SourceFunction, (uint)offset, (uint)func_text.Length, func.Sections));
            offset += func_text.Length;
        }

        return new LinkedModule(linkedFunctions, text.ToArray(), _rdata);
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
