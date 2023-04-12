// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Xml.Schema;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitRandomNormal(in NodeProto op)
        {
            var dtype = GetIntAttribute(op, "dtype");
            var mean = GetFloatAttribute(op, "mean", 0.0f);
            var scale = GetFloatAttribute(op, "scale", 1.0f);
            var seed = GetFloatAttribute(op, "seed", float.NaN);
            var shape = Tensor.From<long>(GetIntsAttribute(op, "shape"));
            return F.Random.Normal(GetDataType(dtype), mean, scale, seed, shape);
        }

        private Expr VisitRandomNormalLike(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var dtype = GetOptionIntAttribute(op, "dtype").
                Match(
                    GetDataType,
                    () => GetDataType(op.Input[0]));
            var mean = GetFloatAttribute(op, "mean", 0.0f);
            var scale = GetFloatAttribute(op, "scale", 1.0f);
            var seed = GetFloatAttribute(op, "seed", float.NaN);
            return F.Random.NormalLike(dtype, input, mean, scale, seed);
        }

        private Expr VisitRandomUniform(in NodeProto op)
        {
            var dtype = GetIntAttribute(op, "dtype");
            var high = GetFloatAttribute(op, "high", 1.0f);
            var low = GetFloatAttribute(op, "low", 0.0f);
            var seed = GetFloatAttribute(op, "seed", float.NaN);
            var shape = Tensor.From<long>(GetIntsAttribute(op, "shape"));
            return F.Random.Uniform(GetDataType(dtype), high, low, seed, shape);
        }

        private Expr VisitRandomUniformLike(NodeProto op)
        {
            var input = GetInputExpr(op, 0);
            var dtype = GetOptionIntAttribute(op, "dtype").
                Match(
                    GetDataType,
                    () => GetDataType(op.Input[0]));
            var high = GetFloatAttribute(op, "high", 1.0f);
            var low = GetFloatAttribute(op, "low", 0.0f);
            var seed = GetFloatAttribute(op, "seed", float.NaN);
            return F.Random.UniformLike(dtype, input, high, low, seed);
        }
    }
}
