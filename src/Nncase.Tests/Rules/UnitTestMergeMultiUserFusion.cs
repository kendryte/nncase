// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Quantization;
using Nncase.Tests.ReWrite.FusionTest;
using Nncase.Tests.TestFixture;
using Nncase.Tests.TransformTest;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Tests.Rules
{
    [AutoSetupTestMethod(InitSession = true)]
    public class UnitTestMergeMultiUserFusion : TransformTestBase
    {
        [Fact]
        public async Task TestSimple()
        {
            var input = Testing.Rand<float>(1, 3, 24, 24);
            var inputVar = new Var(new TensorType(input.ElementType, input.Shape));

            var callee = MakeSingleSimpleFusionCall(Abs, inputVar);
            var caller0 = MakeSingleSimpleFusionCall(Sqrt, callee);
            var caller1 = MakeSingleSimpleFusionCall(Ceil, callee);
            var output = new IR.Tuple(caller0, caller1);
            var dict = new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } };
            await RunTest(output, new[] { inputVar }, dict);
        }

        /// <summary>
        ///
        /// </summary>
        [Fact]
        public async Task TestHasSameInput()
        {
            // tr = transpose(input)
            // callee = Abs(tr)
            // callee + tr | callee - tr
            var input = Testing.Rand<float>(1, 3, 24, 24);
            var inputVar = new Var(new TensorType(input.ElementType, input.Shape));
            var tr = Transpose(inputVar, new[] { 3, 2, 1, 0 });
            var callee = MakeSingleSimpleFusionCall(Abs, tr);
            var caller0 = MakeSimpleFusionCall(args => args[0] + args[1], callee, tr);
            var caller1 = MakeSimpleFusionCall(args => args[0] - args[1], callee, tr);
            var output = new IR.Tuple(caller0, caller1);
            var dict = new Dictionary<Var, IValue> { { inputVar, Value.FromTensor(input) } };
            await RunTest(output, new[] { inputVar }, dict);
        }


        private static Expr GetModuleEntryBody(IRModule module)
        {
            return ((Function)module.Entry!).Body;
        }

        private static async Task RunTest(Expr body, Var[] inputVar, Dictionary<Var, IValue> dict)
        {
            var module = MakeModule(body, inputVar);
            DumpScope.Current.DumpIR(module.Entry!, "origin");
            var preResult = body.Evaluate(dict);
            var preHash = body.GetHashCode();
            var post = await new MergeBucketFusion().RunAsync(module, new());
            DumpScope.Current.DumpIR(post.Entry!, "post");
            Assert.NotEqual(body.GetHashCode(), preHash);
            var postResult = body.Evaluate(dict);
            if (!Comparator.AllEqual(preResult, postResult))
            {
                Comparator.Compare(preResult, postResult);
            }

            var visitor = new FusionCounterVisitor();
            visitor.Visit(body);
            Assert.Equal(1, visitor.Count);
        }

        private static IRModule MakeModule(Expr output, Var[] inputVar) => new(new Function("main", output, inputVar));

        private static Call MakeSingleSimpleFusionCall(Func<Expr, Expr> ctor, Expr arg)
        {
            var v = new Var(arg.CheckedType);
            var abs = ctor(v);
            var f = new BucketFusion("stackvm", abs, new[] { v }, new Var[] { });
            var c = new Call(f, arg);
            return c;
        }

        private static Call MakeSimpleFusionCall(Func<Expr[], Expr> ctor, params Expr[] args)
        {
            var paramList = args.Select(x => new Var(x.CheckedType)).ToArray();
            var abs = ctor(paramList);
            var f = new BucketFusion("stackvm", abs, paramList, new Var[] { });
            var c = new Call(f, args);
            return c;
        }
    }
}
