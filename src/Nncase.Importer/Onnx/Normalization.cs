// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr BroadCastValueByChannel(Expr v)
        {
            return F.Tensors.Broadcast(
                v, F.Tensors.Concat(
                    new IR.Tuple(F.Tensors.ShapeOp(v), new[] { 1 }, new[] { 1 }), 0));
        }
        
        private Expr VisitBatchNormalization(in NodeProto op)
        {
            var x = GetInputExpr(op, 0);
            var (scale, b) = GetInputExprs(op, 1, 2);
            var (mean, var) = GetInputExprs(op, 3, 4);
            var eps = GetFloatAttribute(op, "epsilon", 1e-05f);
            var mom = GetFloatAttribute(op, "momentum", 0.9f);
            var input_mean = BroadCastValueByChannel(mean);
            var bias = BroadCastValueByChannel(b);
            return (x - input_mean) / F.Math.Sqrt(var + eps) * scale + bias;
        }

        private Expr VisitInstanceNormalization(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var (scale, bias) = GetInputExprs(op, 1, 2);
            var eps = GetFloatAttribute(op, "epsilon", 1e-05f);
            return F.NN.InstanceNormalization(input, eps) * scale + bias;
        }

        private Expr VisitLpNormalization(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var axis = GetIntAttribute(op, "axis", -1);
            var p = GetIntAttribute(op, "p", 2);
            return F.NN.LpNormalization(input, axis, p);
        }

        private Expr VisitLRN(in NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var alpha = GetFloatAttribute(op, "alpha", 0.0001f);
            var beta = GetFloatAttribute(op, "beta", 0.75f);
            var bias = GetFloatAttribute(op, "bias", 1.0f);
            var size = GetIntAttribute(op, "size");
            return F.NN.LRN(input, alpha, beta, bias, size);
        } 
    }
}