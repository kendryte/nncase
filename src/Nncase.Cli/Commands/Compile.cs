// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using Autofac;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.CodeGen;
using Nncase.Compiler;
using Nncase.IR;
using Nncase.Transform;

namespace Nncase.Cli.Commands
{
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
            AddArgument(new Argument("output-file"));
            AddOption(new Option<string>(new[] { "-t", "--target" }, "target architecture, e.g. cpu, k210") { IsRequired = true });
            AddOption(new Option<string>(new[] { "-i", "--input-format" }, "input format, e.g. tflite") { IsRequired = true });
            AddOption(new Option<int>("--dump-level", () => 0, "dump ir to .il, default is 0") { IsRequired = false });
            AddOption(new Option<string>("--dump-dir", () => ".", "dump to directory, default is .") { IsRequired = false });

            Handler = CommandHandler.Create<CompileOptions, IHost>(Run);
        }

        internal void ConfigureServices(IHost host)
        {
            var provider = host.Services.GetRequiredService<ICompilerServicesProvider>();
            CompilerServices.Configure(provider);
        }

        private void Run(CompileOptions options, IHost host)
        {
            new Compiler.Compiler().Compile(options, null);
        }
    }
}
