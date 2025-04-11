// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen;

public enum FunctionIdComponent
{
    ModuleId,
    FunctionId,
}

public interface ILinkableFunction
{
    uint Id { get; }

    BaseFunction SourceFunction { get; }

    IEnumerable<FunctionRef> FunctionRefs { get; }

    Stream Text { get; }

    IReadOnlyList<ILinkedSection> Sections { get; }
}

public record FunctionRef(long Position, int Length, BaseFunction Callable, FunctionIdComponent Component, int Offset);
