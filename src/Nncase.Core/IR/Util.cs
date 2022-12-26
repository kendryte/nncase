// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR
{
    public class Util
    {
        public static int PositiveIndex(int index, TensorType input)
        {
            return PositiveIndex(index, input.Shape.Rank);
        }

        public static int PositiveIndex(int index, int rank)
        {
            return index < 0 ? index + rank : index;
        }
    }
}
