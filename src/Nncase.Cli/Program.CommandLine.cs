// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.CommandLine;
using System.CommandLine.Builder;
using System.Linq;

namespace Nncase.Cli;

internal partial class Program
{
    private static CommandLineBuilder BuildCommandLine()
    {
        var commands = from t in typeof(Program).Assembly.ExportedTypes
                       where t.Namespace == "Nncase.Cli.Commands" && t.IsAssignableTo(typeof(Command))
                       select (Command)Activator.CreateInstance(t)!;
        var root = new RootCommand();
        foreach (var command in commands)
        {
            root.AddCommand(command);
        }

        return new CommandLineBuilder(root);
    }
}
