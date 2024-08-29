// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using static Nncase.ResizeModeHelper;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitResize(in NodeProto op)
        {
            var opSet = GetOpSet(op);
            if (opSet <= 10)
            {
                return ResizeV10(op);
            }
            else
            {
                return ResizeV11(op);
            }
        }

        private Expr GetRoi(NodeProto op)
        {
            return GetOptionInputExpr(op, 1).Or(Array.Empty<float>());
        }

        private Expr ResizeV10(in NodeProto op)
        {
            var (input, scales) = GetInputExprs(op, 0, 1);
            var mode = GetResizeMode(op);
            return F.Imaging.ResizeImage(mode, input, Array.Empty<float>(), ComputeNewSizes(input, scales));
        }

        private Expr ResizeV11(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var roi = GetRoi(op);
            var newSize = GetNewSize(op);
            var transformationMode = ParseImageResizeTransformationMode(
                GetStringAttribute(op, "coordinate_transformation_mode", "half_pixel"));
            var nearestMode = ParseImageResizeNearestMode(
                GetStringAttribute(op, "nearest_mode", "round_prefer_floor"));
            var mode = GetResizeMode(op);
            var cubicCoeffA = GetFloatAttribute(op, "cubic_coeff_a", -0.75f);
            var excludeOutside = GetBoolAttribute(op, "exclude_outside", false);
            var extrapolationValue = GetFloatAttribute(op, "extrapolation_value", -0.0f);
            return F.Imaging.ResizeImage(
                mode,
                transformationMode,
                nearestMode,
                input,
                roi,
                newSize,
                cubicCoeffA,
                excludeOutside,
                extrapolationValue);
        }

        private ImageResizeMode GetResizeMode(NodeProto op)
        {
            return ParseResizeMode(GetStringAttribute(op, "mode", "nearest"));
        }

        private Expr ComputeNewSizes(Expr input, Expr scales)
        {
            return Reshape(
                Cast(Cast(ShapeOf(input), DataTypes.Float32) * Unsqueeze(scales, new[] { 0 }), DataTypes.Int64),
                new[] { -1 });
        }

        private Expr GetNewSize(NodeProto op)
        {
            // Only one of 'scales' and 'sizes' can be specified.
            // 2:scales, 3:sizes
            var scales = GetOptionInputExpr(op, 2);
            if (scales.IsSome)
            {
                var scalesValue = scales.ValueUnsafe();
                scalesValue.InferenceType();
                if (scalesValue.CheckedShape[0] != 0 && !scalesValue.CheckedShape.IsScalar)
                {
                    return ComputeNewSizes(GetInputExpr(op, 0), scalesValue);
                }
            }

            var sizes = GetOptionInputExpr(op, 3).ValueUnsafe();
            return sizes;
        }
    }
}
