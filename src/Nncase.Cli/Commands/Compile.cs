// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Cli.Commands
{
    /// <summary>
    /// Options of compile command.
    /// </summary>
    public class CompileOptions
    {
        /// <summary>
        /// Gets or sets input file.
        /// </summary>
        public string InputFile { get; set; }

        /// <summary>
        /// The import model format
        /// </summary>
        public string InputFormat { get; set; }
        /// <summary>
        /// Gets or sets target.
        /// </summary>
        public string Target { get; set; }

        public bool DumpIr { get; set; }

        public string DumpDir { get; set; }
    }

    /// <summary>
    /// Compile command.
    /// </summary>
    public class Compile : Command
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Compile"/> class.
        /// </summary>
        public Compile()
            : base("compile")
        {
            AddArgument(new Argument("input-file"));
            AddOption(new Option<string>(new[] { "-t", "--target" }, "target architecture, e.g. cpu, k210") { IsRequired = true });
            AddOption(new Option<string>(new[] { "-i", "--input-format" }, "input format, e.g. tflite") { IsRequired = true });
            AddOption(new Option<bool>("--dump-ir", () => false, "dump ir to .dot, default is False") { IsRequired = false });
            AddOption(new Option<string>("--dump-dir", () => ".", "dump to directory, default is .") { IsRequired = false });

            Handler = CommandHandler.Create<CompileOptions>(Run);
        }

        public void Run(CompileOptions options)
        {
            Console.WriteLine($"Target: {options.Target}");
            var module = ImportModel(options);
            DumpModule(module, "ir_import");

            var pmgr = new PassManager(module.Entry, new RunPassOptions(null, options.DumpIr, options.DumpDir));
            pmgr.Add(new EGraphPass("eoptimize"));
            pmgr.Run();
        }

        public Module ImportModel(CompileOptions options) =>
          ImportModel(File.OpenRead(options.InputFile), options);

        public Module ImportModel(Stream content, CompileOptions options) =>
          options.InputFormat switch
          {
              "tflite" => Importers.ImportTFLite(content),
              _ => throw new NotImplementedException($"Not Implement {options.InputFormat} Impoter!")
          };

        private void DumpModule(Module module, string prefix)
        {
            var dumpPath = Path.Combine("dump", prefix);
            Directory.CreateDirectory(dumpPath);

            var func = module.Entry;
            using var dumpFile = File.Open(Path.Combine(dumpPath, $"{func.Name}.il"), FileMode.Create);
            using var dumpWriter = new StreamWriter(dumpFile);
            IRPrinter.DumpFunctionAsIL(dumpWriter, func);
        }
    }
}
