// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.CodeGen.Ncnn;

internal class NcnnLinkableFunction : ILinkableFunction
{
    public NcnnLinkableFunction(uint id, BaseFunction sourceFunction, IEnumerable<FunctionRef> functionRefs, Stream text, string[] inputs, string[] outputs)
    {
        Id = id;
        SourceFunction = sourceFunction;
        FunctionRefs = functionRefs;
        Text = text;
        Inputs = inputs;
        Outputs = outputs;
        Sections = new[]
        {
            LinkedSection.FromStrings(inputs, ".inputs"),
            LinkedSection.FromStrings(outputs, ".outputs"),
        };
    }

    public uint Id { get; }

    public BaseFunction SourceFunction { get; }

    public IEnumerable<FunctionRef> FunctionRefs { get; }

    public Stream Text { get; }

    public string[] Inputs { get; }

    public string[] Outputs { get; }

    public IReadOnlyList<ILinkedSection> Sections { get; }
}
