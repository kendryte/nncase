// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    // public partial class OnnxImporter
    // {
    //     private Expr VisitResize(in NodeProto op)
    //     {
    //         var opSet = GetOpSet(op);
    //         if (opSet <= 10)
    //         {
    //             return ResizeV10(op);
    //         }
    //         else if (opSet == 11)
    //         {
    //             return ResizeV11(op);
    //         }
    //         else
    //         {
    //             return ResizeV13(op);
    //         }
    //     }
    //
    //     private ImageResizeMode ParseResizeMode(string mode)
    //     {
    //         return mode switch
    //         {
    //             "nearset" => ImageResizeMode.NearestNeighbor,
    //             "bilinear" => ImageResizeMode.Bilinear,
    //             "trilinear" => ImageResizeMode.Trilinear,
    //             _ => throw new NotSupportedException($"Unsupported Resize Mode {mode}"),
    //         };
    //     }
    //
    //     private ImageResizeMode GetResizeMode(NodeProto op)
    //     {
    //         return ParseResizeMode(GetStringAttribute(op, "mode", "nearest"));
    //     }
    //
    //     private Expr ResizeV10(in NodeProto op)
    //     {
    //         var (input, scales) = GetInputExprs(op, 0, 1);
    //         var mode = GetResizeMode(op);
    //         return F.Tensors.ResizeImage(mode, input,, false, false);
    //     }
    //
    //     private Expr ResizeV11(in NodeProto op)
    //     {
    //         var input = GetInputExpr(op, 0);
    //         var (roi, scales) = GetInputExprs(op, 1, 2);
    //         var sizes = GetOptionInputExpr(op, 3);
    //         var mode = GetResizeMode(op);
    //
    //         var coordinateTransformationMode = GetStringAttribute(op, "coordinate_transformation_mode", "half_pixel");
    //         var cubicCoeffA = GetFloatAttribute(op, "cubic_coeff_a", -0.75f);
    //         var excludeOutside = GetBoolAttribute(op, "exclude_outside", false);
    //         var extrapolationValue = GetFloatAttribute(op, "extrapolation_value", -0.0f);
    //         var nearestMode = GetStringAttribute(op, "nearest_mode", "round_prefer_floor");
    //     }
    //
    //     private Expr ResizeV13(in NodeProto op)
    //     {
    //         var input = GetInputExpr(op, 0);
    //         var (roi, scales) = GetOptionInputExprs(op, 1, 2);
    //         var sizes = GetOptionInputExpr(op, 3);
    //         var mode = GetResizeMode(op);
    //     }
    // }
}