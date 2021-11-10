// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Expr VisitResizeImage(in tflite.Operator op, ImageResizeMode resizeMode)
        {
            var (input, newSize) = GetInputExprs(op, 0, 1);
            var (alignCorners, halfPixelCenters) = GetResizeOptions(op);
            return Util.NCHWToNHWC(
                F.Tensors.ResizeImage(
                    resizeMode, Util.NHWCToNCHW(input), newSize, alignCorners, halfPixelCenters)); 
        }

        private (bool, bool) GetResizeOptions(in tflite.Operator op)
        {
            if (op.BuiltinOptionsType == tflite.BuiltinOptions.ResizeBilinearOptions)
            {
                return (op.BuiltinOptionsAsResizeBilinearOptions().AlignCorners,
                    op.BuiltinOptionsAsResizeBilinearOptions().HalfPixelCenters);
            }
            else if (op.BuiltinOptionsType == tflite.BuiltinOptions.ResizeNearestNeighborOptions)
            {
                return (op.BuiltinOptionsAsResizeNearestNeighborOptions().AlignCorners,
                    op.BuiltinOptionsAsResizeNearestNeighborOptions().HalfPixelCenters);
            }
            else
            {
                throw new NotSupportedException("Unsupported Tflite Option type in resize");
            }
        }
    }
}