// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase;

public struct QuantConfig : IEquatable<QuantConfig>
{
    // [sizeof(input) + sizeof(output)][datatype, min, max]
    private readonly (Tensor<float>, DataType)[] _config = Array.Empty<(Tensor<float>, DataType)>();

    public QuantConfig((Tensor<float>, DataType)[] config)
    {
        _config = config;
    }

    // todo: 但是bychannel的range就不行了吧
    public static QuantConfig FromRaw(Tensor rawTensor)
    {
        var raw = rawTensor.ToArray<float>();
        var config = default(QuantConfig);
        for (int i = 0; i < raw.Length / 3; i++)
        {
            var rawBase = i * 3;
            config._config[i] = (Tensor.From(new[] { raw[rawBase], raw[rawBase + 1] }), QuantTypeMapReverse[(int)raw[rawBase] + 2]);
        }

        return config;
    }

    public Tensor ToRaw()
    {
        return _config.SelectMany(x => x.Item1.ToArray().Append(QuantTypeMap[x.Item2])).ToArray();
    }

    private static readonly Dictionary<DataType, int> QuantTypeMap = new Dictionary<DataType, int>
    {
        { DataTypes.UInt8, 0 }, { DataTypes.Int8, 1 }, { DataTypes.Int16, 2 },
    };

    private static readonly Dictionary<int, DataType> QuantTypeMapReverse = QuantTypeMap.ToDictionary(x => x.Value, x => x.Key);

    public Tensor<float> GetInputRange(ParameterInfo info)
    {
        return _config[info.Index].Item1;
    }

    public DataType GetQuantType(ParameterInfo info)
    {
        return _config[info.Index].Item2;
    }

    // todo: for multi outputs
    public Tensor<float> GetOutputRange()
    {
        return _config[^1].Item1;
    }

    public Tensor<float>[] GetOutputRanges()
    {
        throw new NotImplementedException();
        // return _outRange;
    }

    public bool Equals(QuantConfig other)
    {
        return _config.Equals(other._config);
    }

    public override bool Equals(object? obj)
    {
        return obj is QuantConfig other && Equals(other);
    }

    public override int GetHashCode()
    {
        return _config.GetHashCode();
    }
}
