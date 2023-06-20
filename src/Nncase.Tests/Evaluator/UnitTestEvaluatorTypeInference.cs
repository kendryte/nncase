// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Utilities.DumpUtility;
using Random = Nncase.IR.F.Random;

namespace Nncase.Tests.EvaluatorTest;

public class UnitTestEvaluatorTypeInference
{
    [Fact]
    public void TestCommonType()
    {
        var actual1 = TypeInference.CommonType(DataTypes.Boolean, DataTypes.Float16);
        var expect1 = new InvalidType($"Inputs DType of if should be same, then: bool, else: f16");
        Assert.Equal(actual1, expect1);

        var actual2 = TypeInference.CommonType(DataTypes.Boolean, DataTypes.Boolean);
        var expect2 = new TensorType(DataTypes.Boolean, Array.Empty<int>());
        Assert.Equal(actual2, expect2);

        var thenType3 = new TensorType(DataTypes.Float32, new Shape(1, 3, 16, 16));
        var elseType3 = new TensorType(DataTypes.Float32, new Shape(1, 3, 16, 16));
        var actual3 = TypeInference.CommonType(thenType3, elseType3);
        var expect3 = thenType3;
        Assert.Equal(actual3, expect3);

        var typeArray1 = new List<IRType>();
        typeArray1.Add(DataTypes.Int8);
        typeArray1.Add(DataTypes.Float16);
        var tupleType1 = new TupleType(typeArray1);
        var actual4 = TypeInference.CommonType(tupleType1, tupleType1);
        Assert.Equal(tupleType1, actual4);

        var typeArray2 = new List<IRType>();
        typeArray2.Add(DataTypes.Int8);
        var tupleType2 = new TupleType(typeArray2);
        var actual5 = TypeInference.CommonType(tupleType1, tupleType2);
        var expect5 = new InvalidType($"tuple Inputs of if should be same count, then: {tupleType1.Count}, else: {@tupleType2.Count}");
        Assert.Equal(expect5, actual5);

        var actual7 = TypeInference.CommonType(tupleType1, thenType3);
        var expect7 = new InvalidType($"Inputs of if should be same IRType Kind, but then:{tupleType1}, else: {thenType3}");
        Assert.Equal(expect7, actual7);
    }
}
