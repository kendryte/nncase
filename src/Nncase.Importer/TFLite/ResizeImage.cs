// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using static Nncase.IR.F.Imaging;
using static Nncase.IR.F.Tensors;

namespace Nncase.Importer.TFLite
{
    public partial class TFLiteImporter
    {
        private Shape MakeResizeSizes(Expr input, RankedShape newSize)
        {
            var newNC = new[] {
                Util.ShapeIndex(input, 0),
                Util.ShapeIndex(input, 1),
            };
            return new RankedShape(newNC.Concat(newSize));
        }

        private Expr VisitResizeImage(in tflite.Operator op, ImageResizeMode resizeMode)
        {
            var (input, newSize) = GetInputExprs<Expr, RankedShape>(op, 0, 1);
            input = NHWCToNCHW(input);
            var tranMode = GetResizeOptions(op);
            var nearestMode = tranMode == ImageResizeTransformationMode.Asymmetric
                ? ImageResizeNearestMode.Floor
                : ImageResizeNearestMode.RoundPreferCeil;
            return NCHWToNHWC(
                ResizeImage(
                    resizeMode,
                    input,
                    Array.Empty<float>(),
                    MakeResizeSizes(input, newSize),
                    tranMode,
                    nearestMode,
                    true));
        }

        private ImageResizeTransformationMode GetResizeOptions(in tflite.Operator op)
        {
            if (op.BuiltinOptionsType == tflite.BuiltinOptions.ResizeBilinearOptions)
            {
                if (op.BuiltinOptionsAsResizeBilinearOptions().AlignCorners)
                {
                    return ImageResizeTransformationMode.AlignCorners;
                }
                else if (op.BuiltinOptionsAsResizeBilinearOptions().HalfPixelCenters)
                {
                    return ImageResizeTransformationMode.HalfPixel;
                }
                else
                {
                    return ImageResizeTransformationMode.Asymmetric;
                }
            }
            else if (op.BuiltinOptionsType == tflite.BuiltinOptions.ResizeNearestNeighborOptions)
            {
                if (op.BuiltinOptionsAsResizeNearestNeighborOptions().AlignCorners)
                {
                    return ImageResizeTransformationMode.AlignCorners;
                }
                else if (op.BuiltinOptionsAsResizeNearestNeighborOptions().HalfPixelCenters)
                {
                    return ImageResizeTransformationMode.HalfPixel;
                }
                else
                {
                    return ImageResizeTransformationMode.Asymmetric;
                }
            }
            else
            {
                throw new NotSupportedException("Unsupported Tflite Option type in resize");
            }
        }
    }
}
