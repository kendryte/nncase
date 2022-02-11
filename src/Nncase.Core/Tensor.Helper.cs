// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public partial class Tensor
{
    /// <summary>
    /// Cast and copy to array.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <returns>Casted array.</returns>
    public T[] ToArray<T>()
        where T : unmanaged, IEquatable<T>
        => Cast<T>().ToArray();

    /// <summary>
    /// Cast to a scalar.
    /// </summary>
    /// <typeparam name="T">Scalar type.</typeparam>
    /// <returns>Casted scalar.</returns>
    public T ToScalar<T>()
      where T : unmanaged, IEquatable<T>
    {
        var tensor = Cast<T>();
        if (tensor.Length != 1)
        {
            throw new InvalidOperationException("This tensor is not a scalar.");
        }

        return tensor[0];
    }

    /// <summary>
    /// Cast to string.
    /// </summary>
    /// <returns>string.</returns>
    public string ToStr() => ElementType switch
    {
        { ElemType: ElemType.String, Lanes: 1 } => Encoding.Default.GetString(BytesBuffer),
        _ => throw new InvalidCastException($"This tensor is not a string!"),
    };
}
