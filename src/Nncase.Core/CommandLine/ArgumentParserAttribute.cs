// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CommandLine;

[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class ArgumentParserAttribute : Attribute
{
    public ArgumentParserAttribute(Type value)
    {
        Value = value;
    }

    public Type Value { get; }
}
