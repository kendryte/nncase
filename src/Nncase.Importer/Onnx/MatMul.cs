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
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 55 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 55 }, new int[] { 56 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 56 }, new int[] { 128 }, new int[] { 2 }, new int[] { 1 });
                var d_a = F.Tensors.Slice(a, new int[] { 128 }, new int[] { 129 }, new int[] { 2 }, new int[] { 1 });
                var e_a = F.Tensors.Slice(a, new int[] { 129 }, new int[] { 321 }, new int[] { 2 }, new int[] { 1 });
                var f_a = F.Tensors.Slice(a, new int[] { 321 }, new int[] { 322 }, new int[] { 2 }, new int[] { 1 });
                var g_a = F.Tensors.Slice(a, new int[] { 322 }, new int[] { 1489 }, new int[] { 2 }, new int[] { 1 });
                var h_a = F.Tensors.Slice(a, new int[] { 1489 }, new int[] { 1490 }, new int[] { 2 }, new int[] { 1 });
                var i_a = F.Tensors.Slice(a, new int[] { 1490 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                a_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_a" };
                b_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_b" };
                c_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_c" };
                d_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_d" };
                e_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_e" };
                f_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_f" };
                g_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_g" };
                h_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_h" };
                i_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_i" };

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 55 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 55 }, new int[] { 56 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 56 }, new int[] { 128 }, new int[] { 0 }, new int[] { 1 });
                var d_b = F.Tensors.Slice(b, new int[] { 128 }, new int[] { 129 }, new int[] { 0 }, new int[] { 1 });
                var e_b = F.Tensors.Slice(b, new int[] { 129 }, new int[] { 321 }, new int[] { 0 }, new int[] { 1 });
                var f_b = F.Tensors.Slice(b, new int[] { 321 }, new int[] { 322 }, new int[] { 0 }, new int[] { 1 });
                var g_b = F.Tensors.Slice(b, new int[] { 322 }, new int[] { 1489 }, new int[] { 0 }, new int[] { 1 });
                var h_b = F.Tensors.Slice(b, new int[] { 1489 }, new int[] { 1490 }, new int[] { 0 }, new int[] { 1 });
                var i_b = F.Tensors.Slice(b, new int[] { 1490 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

                a_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_a" };
                b_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_b" };
                c_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_c" };
                d_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_d" };
                e_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_e" };
                f_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_f" };
                g_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_g" };
                h_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_h" };
                i_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_i" };

                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);
                var new_d = F.Math.MatMul(d_a, d_b);
                var new_e = F.Math.MatMul(e_a, e_b);
                var new_f = F.Math.MatMul(f_a, f_b);
                var new_g = F.Math.MatMul(g_a, g_b);
                var new_h = F.Math.MatMul(h_a, h_b);
                var new_i = F.Math.MatMul(i_a, i_b);

                List<string> outputNames_a = new() { op.Output[0] + "_a" };
                List<string> outputNames_b = new() { op.Output[0] + "_b" };
                List<string> outputNames_c = new() { op.Output[0] + "_c" };
                List<string> outputNames_d = new() { op.Output[0] + "_d" };
                List<string> outputNames_e = new() { op.Output[0] + "_e" };
                List<string> outputNames_f = new() { op.Output[0] + "_f" };
                List<string> outputNames_g = new() { op.Output[0] + "_g" };
                List<string> outputNames_h = new() { op.Output[0] + "_h" };
                List<string> outputNames_i = new() { op.Output[0] + "_i" };
                new_a.Metadata.OutputNames = outputNames_a;
                new_b.Metadata.OutputNames = outputNames_b;
                new_c.Metadata.OutputNames = outputNames_c;
                new_d.Metadata.OutputNames = outputNames_d;
                new_e.Metadata.OutputNames = outputNames_e;
                new_f.Metadata.OutputNames = outputNames_f;
                new_g.Metadata.OutputNames = outputNames_g;
                new_h.Metadata.OutputNames = outputNames_h;
                new_i.Metadata.OutputNames = outputNames_i;

                var add_0 = F.Math.Add(new_d, new_e);
                add_0.Metadata.OutputNames = new[] { op.Output[0] + "_add_0" };
                var add_1 = F.Math.Add(add_0, new_c);
                add_1.Metadata.OutputNames = new[] { op.Output[0] + "_add_1" };
                var add_2 = F.Math.Add(add_1, new_b);
                add_2.Metadata.OutputNames = new[] { op.Output[0] + "_add_2" };
                var add_3 = F.Math.Add(add_2, new_a);
                add_3.Metadata.OutputNames = new[] { op.Output[0] + "_add_3" };
                var add_4 = F.Math.Add(add_3, new_f);
                add_4.Metadata.OutputNames = new[] { op.Output[0] + "_add_4" };
                var add_5 = F.Math.Add(add_4, new_g);
                add_5.Metadata.OutputNames = new[] { op.Output[0] + "_add_5" };
                var add_6 = F.Math.Add(add_5, new_h);
                add_6.Metadata.OutputNames = new[] { op.Output[0] + "_add_6" };
                var add_7 = F.Math.Add(add_6, new_i);
                add_7.Metadata.OutputNames = new[] { op.Output[0] + "_add_7" };

                return add_7;
            }
            else if (a.Metadata.OutputNames![0] == "/mlp_26/Mul_output_0")
            {
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 1564 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 1564 }, new int[] { 1565 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 1565 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                a_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_a" };
                b_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_b" };
                c_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_c" };

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 1564 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 1564 }, new int[] { 1565 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 1565 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

                a_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_a" };
                b_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_b" };
                c_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_c" };

                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);

                List<string> outputNames_a = new() { op.Output[0] + "_a" };
                List<string> outputNames_b = new() { op.Output[0] + "_b" };
                List<string> outputNames_c = new() { op.Output[0] + "_c" };
                new_a.Metadata.OutputNames = outputNames_a;
                new_b.Metadata.OutputNames = outputNames_b;
                new_c.Metadata.OutputNames = outputNames_c;

                var add_0 = F.Math.Add(new_c, new_b);
                add_0.Metadata.OutputNames = new[] { op.Output[0] + "add_0" };
                var add_1 = F.Math.Add(new_a, add_0);
                add_1.Metadata.OutputNames = new[] { op.Output[0] + "add_1" };

                return add_1;
            }
            else if (a.Metadata.OutputNames![0] == "/mlp_27/Mul_output_0")
            {
                var splitPoints = new[] { 0, 47, 48, 194, 195, 198, 199, 400, 401, 1551, 1552, 1974, 1975, 1995, 1996, 2038, 2039, 2208, 2209, 2211, 2212, 2326, 2327, 2520, 2521, 2601, 2602, 2842, 2843, -1 };
                var newResults = new List<Call>();

                for (int i = 0; i < splitPoints.Length - 1; i++)
                {
                    var start = new[] { splitPoints[i] };
                    var end = new[] { splitPoints[i + 1] };

                    var a_slice = F.Tensors.Slice(a, start, end, new[] { 2 }, new[] { 1 });
                    var b_slice = F.Tensors.Slice(b, start, end, new[] { 0 }, new[] { 1 });

                    string suffix = ((char)('a' + i)).ToString();
                    a_slice.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_" + suffix };
                    b_slice.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_" + suffix };

                    var result = F.Math.MatMul(a_slice, b_slice);
                    result.Metadata.OutputNames = new[] { op.Output[0] + "_" + suffix };
                    newResults.Add(result);
                }

                // 累加所有 newResults 的值
                var sum = newResults[0];
                for (int i = 1; i < newResults.Count; i++)
                {
                    sum = F.Math.Add(sum, newResults[i]);
                    sum.Metadata.OutputNames = new[] { op.Output[0] + "_add_" + (i - 1) };
                }

                return sum;
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