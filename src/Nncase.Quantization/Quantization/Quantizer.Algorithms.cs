// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Quantization;

internal partial class Quantizer
{
    private static ValueRange<float> GetMinMax(Tensor<float> tensor)
    {
        var buffer = tensor.Buffer;
        var min = float.MaxValue;
        var max = float.MinValue;

        foreach (var value in buffer)
        {
            if (float.IsFinite(value))
            {
                min = Math.Min(min, value);
                max = Math.Max(max, value);
            }
        }

        return new ValueRange<float>(min, max);
    }
}
