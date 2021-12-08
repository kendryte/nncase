// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using LanguageExt.UnsafeValueAccess;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    // public partial class OnnxImporter
    // {
    //     private Expr VisitSplit(in NodeProto op)
    //     {
    //         var opSet = GetOpSet(op);
    //         if(opSet <= 13)
    //         {
    //             return SplitV11(op);
    //         }
    //         else
    //         {
    //             return SplitV13(op);
    //         }
    //     }
    //
    //     private Expr SplitV11(in NodeProto op)
    //     {
    //         var input = GetInputExpr(op, 0);
    //         var outputSize = op.Output.Count;
    //         var axis = GetIntAttribute(op, "axis", 0);
    //         // inShape[axis] / outputSize
    //         var split = GetIntsAttribute(op, "split");
    //         return F.Tensors.Split(input, axis, split);
    //     }
    //
    //     private Expr SplitV13(in NodeProto op)
    //     {
    //         var input = GetInputExpr(op, 0);
    //         var axis = GetIntAttribute(op, "axis", 0);
    //         var split = GetOptionInputExpr(op, 1).Or();
    //         return F.Tensors.Split(input, axis, split);
    //     }
    // }
}