// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase;

public struct QuantConfig : IEquatable<QuantConfig>
{
    private readonly List<QuantConfigData> _inputConfig = new();
    private readonly List<QuantConfigData> _outputConfig = new();

    private QuantConfigHeader _header = new(0, 0);

    public QuantConfig()
    {
    }

    // 没有实际意义，只是为了提供结构，构造空对象的时候可以区分，用于egraph搜索的时候不会被归结到同一个eclass。
    public QuantConfig(int identity)
    {
        _header = new(identity, identity);
    }

    public static QuantConfig FromRaw(Tensor rawTensor)
    {
        // header: sizeof input, sizeof output
        // data: [sizeof(input) + sizeof(output)][datatype, n, min, max]
        var raw = rawTensor.ToArray<float>();
        var config = new QuantConfig();
        config._header = QuantConfigHeader.FromRaw(raw);
        var rawData = raw.AsSpan().Slice(2);
        var rawDataBaseIndex = 0;
        var i = 0;
        while (rawDataBaseIndex < rawData.Length)
        {
            var currentConfig = QuantConfigData.FromRaw(rawData.Slice(rawDataBaseIndex));
            if (i < config._header.SizeOfInput)
            {
                config._inputConfig.Add(currentConfig);
            }
            else
            {
                config._outputConfig.Add(currentConfig);
            }

            rawDataBaseIndex += currentConfig.RawLength;
            i++;
        }

        return config;
    }

    public Tensor ToRaw()
    {
        var header = _header.ToRaw();
        var rawData = _inputConfig.ToArray().Concat(_outputConfig).SelectMany(x => x.ToRaw());
        return header.Concat(rawData).ToArray();
    }

    public Tensor<float> GetInputRange(ParameterInfo info)
    {
        return _inputConfig[info.Index].Range;
    }

    public DataType GetInputQuantType(ParameterInfo info)
    {
        return _inputConfig[info.Index].DType;
    }

    public int GetInputNum()
    {
        return _inputConfig.Count;
    }

    public Tensor<float> GetOutputRange(int outIndex = 0)
    {
        return _outputConfig[outIndex].Range;
    }

    public DataType GetOutputQuantType(int outIndex = 0)
    {
        return _outputConfig[outIndex].DType;
    }

    public int GetOutputNum()
    {
        return _outputConfig.Count;
    }

    public override bool Equals(object? obj)
    {
        return obj is QuantConfig other && Equals(other);
    }

    public bool Equals(QuantConfig other) => _inputConfig.Equals(other._inputConfig) && _outputConfig.Equals(other._outputConfig) && _header.Equals(other._header);

    public bool IsEmpty() => _header == null;

    public override int GetHashCode() => HashCode.Combine(_inputConfig, _outputConfig, _header);
}

public record QuantConfigHeader(int SizeOfInput, int SizeOfOutput)
{
    public float[] ToRaw() => new float[] { SizeOfInput, SizeOfOutput };

    public static QuantConfigHeader FromRaw(Span<float> raw)
    {
        return new QuantConfigHeader((int)raw[0], (int)raw[1]);
    }
}

public record QuantConfigData(Tensor<float> Range, DataType DType)
{
    internal static readonly Dictionary<DataType, int> QuantTypeMap = new Dictionary<DataType, int>
    {
        { DataTypes.UInt8, 0 }, { DataTypes.Int8, 1 }, { DataTypes.Int16, 2 }, { DataTypes.Float16, 3 },
    };

    internal static readonly Dictionary<int, DataType> QuantTypeMapReverse = QuantTypeMap!.ToDictionary(x => x.Value, x => x.Key);

    public int RawLength => Range.Length + 2;

    public int Channels => Range.Length / 2;

    public float[] ToRaw() => Range.ToArray().Prepend(Channels).Prepend((float)QuantTypeMap[DType]).ToArray();

    public static QuantConfigData FromRaw(Span<float> rawData)
    {
        var dt = QuantTypeMapReverse[(int)rawData[0]];
        var channels = (int)rawData[1];
        var range = Tensor.From(rawData.Slice(2, channels * 2).ToArray());
        return new QuantConfigData(range, dt);
    }
}
