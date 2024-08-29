// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.CommandLine;

[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class FromAmongAttribute : Attribute
{
    public FromAmongAttribute(params object[] values)
    {
        Values = values;
    }

    public object[] Values { get; }
}
