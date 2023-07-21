// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
#pragma warning disable
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// StackVM function builder.
/// </summary>
internal class FunctionBuilder : IDisposable
{
    private readonly uint _id;
    private readonly MemoryStream _textContent = new MemoryStream();
    private readonly BinaryWriter _textWriter;
    private readonly BinaryWriter _rdataWriter;

    public FunctionBuilder(uint id, BinaryWriter rdataWriter)
    {
        _id = id;
        _textWriter = new BinaryWriter(_textContent, Encoding.UTF8, leaveOpen: true);
        _rdataWriter = rdataWriter;
    }

    public unsafe LinkableFunction Build(TIR.PrimFunction function)
    {
        // 1. convert func to csource
        var visitor = new CSourceConvertVisitor();
        visitor.Visit(function);
        var functionCSource = visitor.GetFunctionCSource();

        // 2. write the rdata
        foreach (var buffer in function.SchedResult.Rdatas)
        {
            var bytes = buffer.Const!.Value.BytesBuffer;
            if ((uint)bytes.Length != buffer.Size)
            {
                throw new InvalidDataException("The Buffer Szie Not Equal!");
            }

            _rdataWriter.Position((uint)buffer.Start);
            _rdataWriter.Write(bytes);
        }

        return new LinkableFunction(_id, function, functionCSource);
    }

    public void Dispose()
    {
    }
}
