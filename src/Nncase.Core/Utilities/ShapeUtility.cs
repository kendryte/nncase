// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.TIR;
using static NetFabric.Hyperlinq.ArrayExtensions;

namespace Nncase.Utilities;

public static class ShapeUtility
{
    public static List<int> FitNcnnShape(List<int> shape, int axis)
    {
        int positive_axis = axis < 0 ? shape.Count + axis : axis;
        var newShape = new List<int> { 1, shape[positive_axis], 1 };
        for (int i = 0; i < positive_axis; i++)
        {
            newShape[0] *= shape[i];
        }

        for (int i = positive_axis + 1; i < shape.Count; i++)
        {
            newShape[2] *= shape[i];
        }

        return newShape;
    }
}
