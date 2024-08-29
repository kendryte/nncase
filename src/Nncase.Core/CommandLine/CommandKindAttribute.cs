// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CommandLine;

public enum CommandKind
{
    Option,
    Argument,
}

[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class CommandKindAttribute : Attribute
{
    public CommandKindAttribute(CommandKind kind)
    {
        Kind = kind;
    }

    public CommandKind Kind { get; }
}
