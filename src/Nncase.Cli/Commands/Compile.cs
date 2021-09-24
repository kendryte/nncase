using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        /// Gets or sets target.
        /// </summary>
        public string Target { get; set; }
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

            Handler = CommandHandler.Create<CompileOptions>(Run);
        }

        private void Run(CompileOptions options)
        {
            Console.WriteLine($"Target: {options.Target}");
        }
    }
}
