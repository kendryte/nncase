using Xunit;
using System.Runtime.CompilerServices;
using System.IO;
using Nncase.Transform;
using Nncase.IR;
using Nncase.Evaluator;

namespace Nncase.Tests
{
    public static class Testing
    {

        /// <summary>
        /// static get tests_outputs path
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        private static string GetNncsaeDumpDirPath([CallerFilePath] string path = null)
        {
            return Path.GetFullPath(Path.Combine(path, "..", "..", "..", "tests_output"));
        }

        public static string GetTestingFilePath([CallerFilePath] string path = null)
        {
            return path;
        }

        /// <summary>
        /// get the nncase `tests_ouput` path
        /// <remarks>
        /// you can set the subPath for get the `xxx/tests_output/subPath`
        /// </remarks>
        /// </summary>
        /// <param name="subDir">sub directory.</param>
        /// <returns> full path string. </returns>
        public static string GetDumpDirPath(string subDir = "")
        {
            var path = GetNncsaeDumpDirPath();
            if (subDir.Length != 0)
            {
                path = Path.Combine(path, subDir);
                if (!Directory.Exists(path))
                {
                    Directory.CreateDirectory(path);
                }
            }
            return path;
        }

        private static readonly DataFlowPass _simplifyPass = new("SimplifyAll");

        internal static Expr Simplify(Expr expr, string member)
        {
            if (_simplifyPass.Rules.Count == 0)
            {
                _simplifyPass.Add(Transform.Rule.SimplifyFactory.SimplifyAdd());
                _simplifyPass.Add(Transform.Rule.SimplifyFactory.SimplifyMul());
                _simplifyPass.Add(new Transform.Rule.FoldConstCall());
            }
            var f = new Function(expr, new Expr[] { });
            var result = f.InferenceType();
            RunPassOptions options = RunPassOptions.Invalid;

            var dumpPath = Path.Combine(GetNncsaeDumpDirPath(), member, "Simplify");
            int dup = 1;
            while (Directory.Exists(dumpPath))
            {
                dumpPath = Path.Combine(GetNncsaeDumpDirPath(), member, $"Simplify_{++dup}");
            }
            options = new RunPassOptions(null, 2, dumpPath);

            return _simplifyPass.Run(f, options).Body;
        }

        public static void AssertExprEqual(Expr lhs, Expr rhs, [CallerMemberName] string member = null)
        {
            var res = Equals(Simplify(lhs, member), Simplify(rhs, member));
            if (!res)
            {
                lhs.DumpExprAsIL("Lhs", Path.Combine(GetNncsaeDumpDirPath(), member));
                rhs.DumpExprAsIL("Rhs", Path.Combine(GetNncsaeDumpDirPath(), member));
            }
            Assert.True(res);
        }
    }

}