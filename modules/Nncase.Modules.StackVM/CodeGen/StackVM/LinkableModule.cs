// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Runtime.StackVM;

namespace Nncase.CodeGen.StackVM;

internal class LinkableModule : ILinkableModule
{
    private const int _textAlignment = 8;

    private readonly byte[] _rdata;
    private readonly IReadOnlyList<LinkableFunction> _functions;

    public LinkableModule(byte[] rdata, IReadOnlyList<LinkableFunction> functions)
    {
        _rdata = rdata;
        _functions = functions;
    }

    public ILinkedModule Link(ILinkContext linkContext)
    {
        var linkedFunctions = new List<LinkedFunction>();
        var text = new MemoryStream();
        using (var bw = new BinaryWriter(text, Encoding.UTF8, true))
        {
            foreach (var func in _functions)
            {
                FixFunctionRefs(func, linkContext);
                bw.AlignPosition(_textAlignment);
                var textBegin = bw.Position();
                bw.Write(func.Text);
                linkedFunctions.Add(new LinkedFunction(func.Id, func.SourceFunction, (uint)textBegin, (uint)func.Text.Length, Array.Empty<ILinkedSection>()));
            }
        }

        return new LinkedModule(linkedFunctions, text.ToArray(), _rdata);
    }

    private void FixFunctionRefs(LinkableFunction func, ILinkContext linkContext)
    {
        using var writer = new BinaryWriter(new MemoryStream(func.Text));
        foreach (var funcRef in func.FunctionRefs)
        {
            var id = linkContext.GetFunctionId(funcRef.Callable);
            var value = funcRef.Component == FunctionIdComponent.ModuleId ? id.ModuleId : id.Id;
            writer.Position(funcRef.Position);
            writer.WriteByLength(value, funcRef.Length);
        }
    }
}
