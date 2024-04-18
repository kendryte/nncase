// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes;

public static class PassUtility
{
    public static bool IsCpuSupported(Op op)
    {
        if (op.GetType().Namespace == "Nncase.IR.CPU")
        {
            return true;
        }

        return op is IR.Math.Unary or IR.Math.MatMul or IR.NN.Conv2D or IR.NN.Softmax or IR.NN.LayerNorm or IR.NN.InstanceNormalization or IR.Imaging.ResizeImage or IR.Tensors.Unsqueeze or IR.Tensors.Reshape or IR.Tensors.Slice or IR.Tensors.Concat or IR.Tensors.Transpose or IR.NN.Swish or IR.Tensors.Gather or IR.NN.Pad;
    }
}
