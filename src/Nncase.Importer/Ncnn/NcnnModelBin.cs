// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace Nncase.Importer.Ncnn;

internal class NcnnModelBin
{
    private readonly Stream _stream;

    public NcnnModelBin(Stream stream)
    {
        _stream = stream;
    }

    public Tensor<float> LoadFloat32(ReadOnlySpan<int> shape, bool detectType)
    {
        if (!detectType)
        {
            var tensor = new Tensor<float>(shape);
            _stream.ReadExactly(tensor.BytesBuffer);
            return tensor;
        }
        else
        {
            return LoadAuto(shape).Cast<float>();
        }
    }

    public Tensor LoadAuto(ReadOnlySpan<int> shape)
    {
        uint tag;
        Unsafe.SkipInit(out tag);
        _stream.ReadExactly(MemoryMarshal.CreateSpan(ref tag, 1).AsBytes());

        if (tag == 0x01306B47)
        {
            // half-precision data
            var tensor = new Tensor<Half>(shape);
            _stream.ReadExactly(tensor.BytesBuffer);
            AlignStream(_stream, tensor.BytesBuffer.Length, 4);
            return tensor.Cast<float>(CastMode.KDefault);
        }
        else if (tag == 0)
        {
            // raw data
            var tensor = new Tensor<float>(shape);
            _stream.ReadExactly(tensor.BytesBuffer);
            return tensor;
        }
        else
        {
            throw new NotSupportedException($"Unsupported weight tag: {tag}.");
        }
    }

    private static void AlignStream(Stream stream, int size, int alignment)
    {
        var rem = size % alignment;
        var offset = rem == 0 ? 0 : alignment - rem;
        stream.Seek(offset, SeekOrigin.Current);
    }
}
