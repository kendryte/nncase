// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase;

public enum MaskVectorStyle
{
    Unknown,
    Fat,
    Slim,
}

public interface IVector<T>
{
    static abstract int Count { get; }
}

public interface IMaskVector : IVector<bool>
{
    void CopyFrom(ReadOnlySpan<bool> values);

    void CopyTo(Span<bool> values);
}
