// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
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

    public LinkableModule(byte[] rdata, IReadOnlyList<LinkableFunction> functions)
    {
        _rdata = rdata;
        _functions = functions;
    }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        var csourcePath = LinkCSources();
        var elfPath = CompileCSource(csourcePath);
        var text = File.ReadAllBytes(elfPath);

        if (DumpScope.Current.IsEnabled(DumpFlags.CodeGen))
        {
            using (var fs = DumpScope.Current.OpenFile("cpuModule.h"))
            {
                using (var writer = new StreamWriter(fs))
                {
                    writer.Write(CSourceBuiltn.Header);
                }
            }

            using (var fs = DumpScope.Current.OpenFile("cpuModule.c"))
            {
                File.Open(csourcePath, FileMode.Open, FileAccess.Read).CopyTo(fs);
            }
        }

        var linkedFunctions = new List<LinkedFunction>();
        foreach (var func in _functions)
        {
            linkedFunctions.Add(new LinkedFunction(func.Id, func.SourceFunction, 0, 0, func.Sections));
        }

        return new LinkedModule(linkedFunctions, text, _rdata);
    }

    private string LinkCSources()
    {
        var path = Path.GetTempFileName();
        using (var fs = File.OpenWrite(path))
        {
            using (var writer = new StreamWriter(fs))
            {
                writer.WriteLine(CSourceBuiltn.Header);

                foreach (var func in _functions)
                {
                    writer.WriteLine(func.FunctionCSource.Declaration);
                }

                foreach (var func in _functions)
                {
                    writer.WriteLine(func.FunctionCSource.Implementation);
                }

                writer.WriteLine(CSourceBuiltn.MainPrologue);
                foreach (var func in _functions)
                {
                    writer.WriteLine($"  if (strcmp(name,\"{func.SourceFunction.Name}\") == 0) {{");
                    writer.WriteLine($"    {func.SourceFunction.Name}({string.Join(",", Enumerable.Range(0, func.PrimFunction.Parameters.Length).Select(i => $"buffers[{i}]"))}, nncase_mt, data, rdata);");
                    writer.WriteLine("  } else");
                }

                writer.WriteLine("  { }");
                writer.WriteLine(CSourceBuiltn.MainEpilogue);
            }
        }

        return path;
    }

    private string CompileCSource(string sourcePath)
    {
        var compiler = new CSourceCompiler();
        return compiler.Compile(sourcePath, Path.GetTempFileName());
    }
}
