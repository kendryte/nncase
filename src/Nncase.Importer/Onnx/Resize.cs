// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;
using static Nncase.ResizeModeHelper;

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
            // todo:string input or attr
            // todo:multi type is mismatch
            // todo:is input is not full, index will be error
            var (input, scales) = GetInputExprs(op, 0, 2);
            var roi = GetRoi(op);
            var mode = GetResizeMode(op);
            return F.Imaging.ResizeImage(mode, input, roi, ComputeNewSizes(input, scales));
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
            return F.Imaging.ResizeImage(mode, transformationMode, nearestMode, input, roi, newSize, cubicCoeffA,
                excludeOutside, extrapolationValue);
        }
        private ImageResizeMode GetResizeMode(NodeProto op)
        {
            return ParseResizeMode(GetStringAttribute(op, "mode", "nearest"));
        }

        private Expr ComputeNewSizes(Expr input, Expr scales)
        {
            return F.Tensors.ShapeOf(input) * scales;
        }
        private Expr GetNewSize(NodeProto op)
        {
            // Only one of 'scales' and 'sizes' can be specified.
            // 2:scales, 3:sizes
            return GetOptionInputExpr(op, 2).Match(
                scales => ComputeNewSizes(GetInputExpr(op, 0), scales),
                GetOptionInputExpr(op, 3).ValueUnsafe());
        }
    }
}