// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase;

public struct QuantConfig : IEquatable<QuantConfig>
{
    private Dictionary<ParameterInfo, (ValueRange<float>, DataType)> _config = new();

    private ValueRange<float>[] _outRange = Array.Empty<ValueRange<float>>();

    public QuantConfig()
    {
    }

    public ValueRange<float> GetRange(ParameterInfo info) => _config[info].Item1;

    public DataType GetQuantType(ParameterInfo info) => _config[info].Item2;

    public bool Equals(QuantConfig other) => _config.Equals(other._config);

    public override bool Equals(object? obj) => obj is QuantConfig other && Equals(other);

    public override int GetHashCode() => _config.GetHashCode();
}

public record QuantConfigType() : ValueType
{
    public override Type CLRType => typeof(QuantConfig);

    public unsafe override int SizeInBytes => sizeof(QuantConfig);

    public override Guid Uuid { get; }
}
