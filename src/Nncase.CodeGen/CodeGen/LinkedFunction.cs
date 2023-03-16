// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

public class LinkedFunction : ILinkedFunction
{
    public LinkedFunction(uint id, Callable sourceFunction, uint textBegin, uint textLength, IReadOnlyList<ILinkedSection> sections)
    {
        Id = id;
        CompilerServices.InferenceType(sourceFunction);
        ParameterTypes = ((CallableType)sourceFunction.CheckedType).Parameters.ToArray();
        ReturnType = ((CallableType)sourceFunction.CheckedType).ReturnType;
        TextBegin = textBegin;
        TextLength = textLength;
        Sections = sections;
    }

    public uint Id { get; }

    public IReadOnlyList<IRType> ParameterTypes { get; }

    public IRType ReturnType { get; }

    public uint TextBegin { get; }

    public uint TextLength { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
