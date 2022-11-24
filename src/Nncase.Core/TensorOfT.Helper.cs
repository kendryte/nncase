// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase;

public partial class Tensor<T>
{
    /// <summary>
    /// Cast and copy to array.
    /// </summary>
    /// <returns>Casted array.</returns>
    public T[] ToArray()
    {
        var array = new T[Length];
        Buffer.CopyTo(array);
        return array;
    }

    /// <summary>
    /// Cast to a scalar.
    /// </summary>
    /// <returns>Casted scalar.</returns>
    public T ToScalar()
    {
        if (Length != 1)
        {
            throw new InvalidOperationException("This tensor is not a scalar.");
        }

        return Buffer.Span[0];
    }
}
