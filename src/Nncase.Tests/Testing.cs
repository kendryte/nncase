// using System.IO;
// using System.Runtime.CompilerServices;
// using System.Threading.Tasks;
// using Autofac;
// using Autofac.Extras.CommonServiceLocator;
// using CommonServiceLocator;
// using Microsoft.Extensions.Configuration;
// using Microsoft.Extensions.DependencyInjection;
// using Microsoft.Extensions.Hosting;
// using Microsoft.Extensions.Logging;
// using Nncase.Evaluator;
// using Nncase.IR;
// using Nncase.Transform;
// using Nncase.Transform.Rules;
// using Xunit;
//
// namespace Nncase.Tests
// {
//     public static class Testing
//     {
//
//          /// <summary>
//          /// static get tests_outputs path
//          /// </summary>
//          /// <param name="path"></param>
//          /// <returns></returns>
//          private static string GetNncsaeDumpDirPath([CallerFilePath] string path = null)
//          {
//              return Path.GetFullPath(Path.Combine(path, "..", "..", "..", "tests_output"));
//          }
//
//          public static string GetTestingFilePath([CallerFilePath] string path = null)
//          {
//              return path;
//          }
//
//          /// <summary>
//          /// get the nncase `tests_ouput` path
//          /// <remarks>
//          /// you can set the subPath for get the `xxx/tests_output/subPath`
//          /// </remarks>
//          /// </summary>
//          /// <param name="subDir">sub directory.</param>
//          /// <returns> full path string. </returns>
//          public static string GetDumpDirPath(string subDir = "")
//          {
//              var path = GetNncsaeDumpDirPath();
//              if (subDir.Length != 0)
//              {
//                  path = Path.Combine(path, subDir);
//                  if (!Directory.Exists(path))
//                  {
//                      Directory.CreateDirectory(path);
//                  }
//              }
//              return path;
//          }
//
//          /// <summary>
//          /// give the unittest class name, then return the dumpdir path
//          /// <see cref="GetDumpDirPath(string)"/>
//          /// </summary>
//          /// <param name="type"></param>
//          /// <returns></returns>
//          public static string GetDumpDirPath(System.Type type)
//          {
//              var namespace_name = type.Namespace.Split(".")[^1];
//              if (!namespace_name.EndsWith("Test") || !type.Name.StartsWith("UnitTest"))
//              {
//                  throw new System.ArgumentOutOfRangeException("We Need NameSpace is `xxxTest`, Class is `UnitTestxxx`");
//              }
//              return GetDumpDirPath(Path.Combine(namespace_name, type.Name));
//          }
//
//          private static readonly DataflowPass _simplifyPass = new("SimplifyAll");
//
//          internal static async Task<Expr> Simplify(Expr expr, string member)
//          {
//              if (_simplifyPass.Rules.Count == 0)
//              {
//                  _simplifyPass.Add(Transform.Rules.Neutral.SimplifyFactory.SimplifyAdd());
//                  _simplifyPass.Add(Transform.Rules.Neutral.SimplifyFactory.SimplifyMul());
//                  // _simplifyPass.Add(new Transform.Rule.FoldConstCall());
//              }
//              var f = new Function(expr, new Var[] { });
//              var result = f.InferenceType();
//              RunPassOptions options = RunPassOptions.Invalid;
//
//              var dumpPath = Path.Combine(GetNncsaeDumpDirPath(), member, "Simplify");
//              int dup = 1;
//              while (Directory.Exists(dumpPath))
//              {
//                  dumpPath = Path.Combine(GetNncsaeDumpDirPath(), member, $"Simplify_{++dup}");
//              }
//              options = new RunPassOptions(null, 2, dumpPath);
//
//              return ((Function)await _simplifyPass.RunAsync(f, options)).Body;
//          }
//
//          public static async Task AssertExprEqual(Expr lhs, Expr rhs, [CallerMemberName] string member = null)
//          {
//              var simpled_lhs = await Simplify(lhs, member);
//              var simpled_rhs = await Simplify(rhs, member);
//              var res = Equals(simpled_lhs, simpled_rhs);
//              if (!res)
//              {
//                  CompilerServices.DumpIR(lhs, "Lhs", Path.Combine(GetNncsaeDumpDirPath(), member));
//                  CompilerServices.DumpIR(rhs, "Rhs", Path.Combine(GetNncsaeDumpDirPath(), member));
//                  CompilerServices.DumpIR(simpled_lhs, "Simpled_Lhs", Path.Combine(GetNncsaeDumpDirPath(), member));
//                  CompilerServices.DumpIR(simpled_rhs, "Simpled_Rhs", Path.Combine(GetNncsaeDumpDirPath(), member));
//              }
//              Assert.True(res);
//          }
//      }
//
//     public abstract class IHostFixtrue
//     {
//         public IHostFixtrue(IHost host)
//         {
//             var t = host.Services.GetRequiredService<IComponentContext>();
//             var csl = new AutofacServiceLocator(t);
//             ServiceLocator.SetLocatorProvider(() => csl);
//         }
//     }
// }