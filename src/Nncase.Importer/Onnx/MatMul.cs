// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer
{
    public partial class OnnxImporter
    {
        private Expr VisitMatMul(in NodeProto op)
        {
            var (a, b) = GetInputExprs(op, 0, 1);
            // /mlp_2/Mul_output_0、/mlp_3/Mul_output_0、/mlp_21/Mul_output_0
            if (a.Metadata.OutputNames![0] == "/mlp_2/Mul_output_0")
            {
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 813 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 813 }, new int[] { 814 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 814 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 813 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 813 }, new int[] { 814 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 814 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });
                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);
                return F.Math.Add(new_a, F.Math.Add(new_c, new_b));
            }
            else if (a.Metadata.OutputNames![0] == "/mlp_3/Mul_output_0")
            {
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 2247 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 2247 }, new int[] { 2248 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 2248 }, new int[] { 3016 }, new int[] { 2 }, new int[] { 1 });
                var d_a = F.Tensors.Slice(a, new int[] { 3016 }, new int[] { 3017 }, new int[] { 2 }, new int[] { 1 });
                var e_a = F.Tensors.Slice(a, new int[] { 3017 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 2247 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 2247 }, new int[] { 2248 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 2248 }, new int[] { 3016 }, new int[] { 0 }, new int[] { 1 });
                var d_b = F.Tensors.Slice(b, new int[] { 3016 }, new int[] { 3017 }, new int[] { 0 }, new int[] { 1 });
                var e_b = F.Tensors.Slice(b, new int[] { 3017 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);
                var new_d = F.Math.MatMul(d_a, d_b);
                var new_e = F.Math.MatMul(e_a, e_b);

                return F.Math.Add(new_a, F.Math.Add(F.Math.Add(F.Math.Add(new_d, new_e), new_c), new_b));
            }
            else if (a.Metadata.OutputNames![0] == "/mlp_21/Mul_output_0")
            {
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 567 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 567 }, new int[] { 568 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 568 }, new int[] { 3486 }, new int[] { 2 }, new int[] { 1 });
                var d_a = F.Tensors.Slice(a, new int[] { 3486 }, new int[] { 3487 }, new int[] { 2 }, new int[] { 1 });
                var e_a = F.Tensors.Slice(a, new int[] { 3487 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 567 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 567 }, new int[] { 568 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 568 }, new int[] { 3486 }, new int[] { 0 }, new int[] { 1 });
                var d_b = F.Tensors.Slice(b, new int[] { 3486 }, new int[] { 3487 }, new int[] { 0 }, new int[] { 1 });
                var e_b = F.Tensors.Slice(b, new int[] { 3487 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);
                var new_d = F.Math.MatMul(d_a, d_b);
                var new_e = F.Math.MatMul(e_a, e_b);

                return F.Math.Add(new_a, F.Math.Add(F.Math.Add(F.Math.Add(new_d, new_e), new_c), new_b));
            }
            else
            {
                var matmul = IR.F.Math.MatMul(a, b);
                List<string> outputNames = new() { op.Output[0] };
                matmul.Metadata.OutputNames = outputNames;
                return matmul;
            }
        }
    }
}