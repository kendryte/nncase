using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Transform.DataFlow.Rules;
using Xunit;
using System.Runtime.CompilerServices;
using System.IO;


namespace Nncase.Tests
{

    public static class Testing
    {

        private static string GetDumpPath([CallerFilePath] string path = null)
        {
            return Path.GetFullPath(Path.Combine(path, "..", "..", "..", "tests_output"));
        }

        private static readonly DataFlowPass _simplifyPass = new("SimplifyAll");

        internal static Expr Simplify(Expr expr, string member)
        {
            if (_simplifyPass.Rules.Count == 0)
            {
                _simplifyPass.Add(SimplifyFactory.SimplifyAdd());
                _simplifyPass.Add(SimplifyFactory.SimplifyMul());
                _simplifyPass.Add(new FoldConstCall());
            }
            var f = new Function(expr, new Expr[] { });
            var result = f.InferenceType();
            RunPassOptions options = RunPassOptions.Invalid;

            var dumpPath = Path.Combine(GetDumpPath(), member, "Simplify");
            int dup = 1;
            while (Directory.Exists(dumpPath))
            {
                dumpPath = Path.Combine(GetDumpPath(), member, $"Simplify_{++dup}");
            }
            options = new RunPassOptions(null, 2, dumpPath);

            return _simplifyPass.Run(f, options).Body;
        }

        public static void AssertExprEqual(Expr lhs, Expr rhs, [CallerMemberName] string member = null)
        {
            var res = Equals(Simplify(lhs, member), Simplify(rhs, member));
            if (!res)
            {
                lhs.DumpExprAsIL("Lhs", Path.Combine(GetDumpPath(), member));
                rhs.DumpExprAsIL("Rhs", Path.Combine(GetDumpPath(), member));
            }
            Assert.True(res);
        }
    }

}