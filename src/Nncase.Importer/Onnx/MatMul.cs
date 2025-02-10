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

                a_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_a" };
                b_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_b" };
                c_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_c" };

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 813 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 813 }, new int[] { 814 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 814 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

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
            else if (a.Metadata.OutputNames![0] == "/mlp_3/Mul_output_0")
            {
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 2247 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 2247 }, new int[] { 2248 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 2248 }, new int[] { 3016 }, new int[] { 2 }, new int[] { 1 });
                var d_a = F.Tensors.Slice(a, new int[] { 3016 }, new int[] { 3017 }, new int[] { 2 }, new int[] { 1 });
                var e_a = F.Tensors.Slice(a, new int[] { 3017 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                a_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_a" };
                b_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_b" };
                c_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_c" };
                d_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_d" };
                e_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_e" };

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 2247 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 2247 }, new int[] { 2248 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 2248 }, new int[] { 3016 }, new int[] { 0 }, new int[] { 1 });
                var d_b = F.Tensors.Slice(b, new int[] { 3016 }, new int[] { 3017 }, new int[] { 0 }, new int[] { 1 });
                var e_b = F.Tensors.Slice(b, new int[] { 3017 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

                a_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_a" };
                b_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_b" };
                c_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_c" };
                d_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_d" };
                e_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_e" };

                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);
                var new_d = F.Math.MatMul(d_a, d_b);
                var new_e = F.Math.MatMul(e_a, e_b);

                List<string> outputNames_a = new() { op.Output[0] + "_a" };
                List<string> outputNames_b = new() { op.Output[0] + "_b" };
                List<string> outputNames_c = new() { op.Output[0] + "_c" };
                List<string> outputNames_d = new() { op.Output[0] + "_d" };
                List<string> outputNames_e = new() { op.Output[0] + "_e" };
                new_a.Metadata.OutputNames = outputNames_a;
                new_b.Metadata.OutputNames = outputNames_b;
                new_c.Metadata.OutputNames = outputNames_c;
                new_d.Metadata.OutputNames = outputNames_d;
                new_e.Metadata.OutputNames = outputNames_e;

                var add_0 = F.Math.Add(new_d, new_e);
                add_0.Metadata.OutputNames = new[] { op.Output[0] + "_add_0" };
                var add_1 = F.Math.Add(add_0, new_c);
                add_1.Metadata.OutputNames = new[] { op.Output[0] + "_add_1" };
                var add_2 = F.Math.Add(add_1, new_b);
                add_2.Metadata.OutputNames = new[] { op.Output[0] + "_add_2" };
                var add_3 = F.Math.Add(add_2, new_a);
                add_3.Metadata.OutputNames = new[] { op.Output[0] + "_add_3" };

                return add_3;
            }
            else if (a.Metadata.OutputNames![0] == "/mlp_21/Mul_output_0")
            {
                var a_a = F.Tensors.Slice(a, new int[] { 0 }, new int[] { 567 }, new int[] { 2 }, new int[] { 1 });
                var b_a = F.Tensors.Slice(a, new int[] { 567 }, new int[] { 568 }, new int[] { 2 }, new int[] { 1 });
                var c_a = F.Tensors.Slice(a, new int[] { 568 }, new int[] { 3486 }, new int[] { 2 }, new int[] { 1 });
                var d_a = F.Tensors.Slice(a, new int[] { 3486 }, new int[] { 3487 }, new int[] { 2 }, new int[] { 1 });
                var e_a = F.Tensors.Slice(a, new int[] { 3487 }, new int[] { -1 }, new int[] { 2 }, new int[] { 1 });

                a_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_a" };
                b_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_b" };
                c_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_c" };
                d_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_d" };
                e_a.Metadata.OutputNames = new[] { a.Metadata.OutputNames[0] + "_e" };

                var a_b = F.Tensors.Slice(b, new int[] { 0 }, new int[] { 567 }, new int[] { 0 }, new int[] { 1 });
                var b_b = F.Tensors.Slice(b, new int[] { 567 }, new int[] { 568 }, new int[] { 0 }, new int[] { 1 });
                var c_b = F.Tensors.Slice(b, new int[] { 568 }, new int[] { 3486 }, new int[] { 0 }, new int[] { 1 });
                var d_b = F.Tensors.Slice(b, new int[] { 3486 }, new int[] { 3487 }, new int[] { 0 }, new int[] { 1 });
                var e_b = F.Tensors.Slice(b, new int[] { 3487 }, new int[] { -1 }, new int[] { 0 }, new int[] { 1 });

                a_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_a" };
                b_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_b" };
                c_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_c" };
                d_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_d" };
                e_b.Metadata.OutputNames = new[] { b.Metadata.OutputNames[0] + "_e" };

                var new_a = F.Math.MatMul(a_a, a_b);
                var new_b = F.Math.MatMul(b_a, b_b);
                var new_c = F.Math.MatMul(c_a, c_b);
                var new_d = F.Math.MatMul(d_a, d_b);
                var new_e = F.Math.MatMul(e_a, e_b);

                List<string> outputNames_a = new() { op.Output[0] + "_a" };
                List<string> outputNames_b = new() { op.Output[0] + "_b" };
                List<string> outputNames_c = new() { op.Output[0] + "_c" };
                List<string> outputNames_d = new() { op.Output[0] + "_d" };
                List<string> outputNames_e = new() { op.Output[0] + "_e" };
                new_a.Metadata.OutputNames = outputNames_a;
                new_b.Metadata.OutputNames = outputNames_b;
                new_c.Metadata.OutputNames = outputNames_c;
                new_d.Metadata.OutputNames = outputNames_d;
                new_e.Metadata.OutputNames = outputNames_e;

                var add_0 = F.Math.Add(new_d, new_e);
                add_0.Metadata.OutputNames = new[] { op.Output[0] + "_add_0" };
                var add_1 = F.Math.Add(add_0, new_c);
                add_1.Metadata.OutputNames = new[] { op.Output[0] + "_add_1" };
                var add_2 = F.Math.Add(add_1, new_b);
                add_2.Metadata.OutputNames = new[] { op.Output[0] + "_add_2" };
                var add_3 = F.Math.Add(add_2, new_a);
                add_3.Metadata.OutputNames = new[] { op.Output[0] + "_add_3" };

                return add_3;
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