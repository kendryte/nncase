// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Nncase.Cli
{
    internal partial class Program
    {
        private static CommandLineBuilder BuildCommandLine()
        {
            var commands = from t in typeof(Program).Assembly.ExportedTypes
                           where t.Namespace == "Nncase.Cli.Commands" && t.IsAssignableTo(typeof(Command))
                           select (Command)Activator.CreateInstance(t);
            var root = new RootCommand();
            foreach (var command in commands)
            {
                root.AddCommand(command);
            }

            return new CommandLineBuilder(root);
        }
    }
}
