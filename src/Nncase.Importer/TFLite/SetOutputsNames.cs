// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using tflite;
using static Nncase.IR.F.NN;
using F = Nncase.IR.F;

namespace Nncase.Importer.TFLite;
public partial class TFLiteImporter
{
    private Expr SetOutputsNames(Expr expr, int outputsNumber, in tflite.Operator op)
    {
        for (int i = 0; i < outputsNumber; i++)
        {
            expr.OutputsNames.Add(GetOutputTensor(op, i).Name);
        }

        return expr;
    }
}
