// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;

namespace System.Collections.Generic;

public sealed class ReferenceEqualityComparer<T> : IEqualityComparer<T>, IEqualityComparer
    where T : class
{
    public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);

    public bool Equals(T? x, T? y) => ReferenceEquals(x, y);

    public int GetHashCode(T? obj)
    {
        // Depending on target framework, RuntimeHelpers.GetHashCode might not be annotated
        // with the proper nullability attribute. We'll suppress any warning that might
        // result.
        return RuntimeHelpers.GetHashCode(obj!);
    }

    public int GetHashCode(object obj) => throw new NotImplementedException();
}
