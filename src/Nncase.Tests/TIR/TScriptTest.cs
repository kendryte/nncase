using Xunit;
using System.Runtime.CompilerServices;
using System.Linq.Expressions;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;
using System;
using static Nncase.IR.F.Math;
using Nncase.TIR.F;
using Nncase.TIR;
using Nncase.IR;

namespace Nncase.Tests.TIRTest
{

    public static class T
    {
        public class AnyExpr
        {
            object value;

            public AnyExpr(object input)
            {
                value = input;
            }
            public static implicit operator AnyExpr(int input) => new AnyExpr(input);
            public static AnyExpr operator +(AnyExpr rhs, AnyExpr lhs) => lhs;
            public static AnyExpr operator -(AnyExpr rhs, AnyExpr lhs) => lhs;
            public static AnyExpr operator *(AnyExpr rhs, AnyExpr lhs) => lhs;
            public static AnyExpr operator /(AnyExpr rhs, AnyExpr lhs) => lhs;
        }

        public class ScopeExpr : AnyExpr, IDisposable
        {
            public ScopeExpr(object input) : base(input) { }
            public void Dispose()
            {
            }
        }

        public class Buffer : AnyExpr
        {
            public Buffer(object input) : base(input) { }
            public AnyExpr this[params AnyExpr[] exprs]
            {
                get => new AnyExpr(null);
                set { }
            }
        }

        public class Handle
        {
        }

        public static ScopeExpr Init(string name = "init") => new ScopeExpr(name);
        public static ScopeExpr Block(string name = "block") => new ScopeExpr(name);

        public static IEnumerable<(AnyExpr, AnyExpr, AnyExpr)> grid(ValueTuple<AnyExpr, AnyExpr, AnyExpr> extents)
        {
            return new[] { extents };
        }
        public static AnyExpr Float32(float input) => new AnyExpr(input);

        public static class Axis
        {
            public static ValueTuple<AnyExpr, AnyExpr, AnyExpr> Remap(string iter_types, ValueTuple<AnyExpr, AnyExpr, AnyExpr> loop_vars)
            {
                return loop_vars;
            }
        }

    }
    public class IRBuilderTest
    {
        public void matmul(T.Handle x, T.Handle y, T.Handle z)
        {
            var A = new T.Buffer(x);
            var B = new T.Buffer(y);
            var C = new T.Buffer(z);
            foreach (var (i, j, k) in T.grid((1, 2, 3)))
            {
                using (T.Block())
                {
                    var (vi, vj, vk) = T.Axis.Remap("SSR", (i, j, k));
                    using (T.Init()) { C[i, j] = T.Float32(0); }
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk];
                }
            }
        }


        [Fact]
        public void TestParser()
        {
            // Expression e = matmul;
        }

        [Fact]
        public void TestExpressionToTIR()
        {
            // TIR.SizeVar n = "n", m = "m";
            var A = TIR.Buffer.Decl((10, 20));
            var B = TIR.Buffer.Decl((20, 30));
            // Expression simple = (int n, int m) => (n + m); TODO how to emit assgin op?
            // var shape = ()
            // var R = TIR.IterVar()
            Expression matmul = (int i, int j, int k) => (A[i, j] * B[j, k]);
        }

        public class TestSimpleVisior : ExprFunctor<string, string>
        {
        }

        [Fact]
        public void TestRelectVisitor()
        {
            var v = new TestSimpleVisior();
            
        }
    }
}