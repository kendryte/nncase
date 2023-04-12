// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.StackVM;

internal class StackVMLinkableFunction : ILinkableFunction
{
    public StackVMLinkableFunction(uint id, BaseFunction sourceFunction, IEnumerable<FunctionRef> functionRefs, ushort maxLocals, byte[] text, IReadOnlySet<ModuleType> custom_call_modules)
    {
        Id = id;
        SourceFunction = sourceFunction;
        FunctionRefs = functionRefs;
        MaxLocals = maxLocals;
        Text = text;
        CustomCallModules = custom_call_modules;
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public IEnumerable<FunctionRef> FunctionRefs { get; }

    public ushort MaxLocals { get; }

    public byte[] Text { get; }

    public IReadOnlyList<ILinkedSection> Sections => Array.Empty<ILinkedSection>();

    public IReadOnlySet<ModuleType> CustomCallModules { get; init; }
}
