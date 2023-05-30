// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.CodeGen
{
    public static class CodeGenDumper
    {
        public static void DumpIdMap(Dictionary<BaseFunction, FunctionId> ids)
        {
            var idInfo = ids.Select(pair => $"{pair.Value.ModuleId} {pair.Value.Id} {pair.Key.Name}").ToArray();
            DumpUtility.WriteResult(Path.Join(DumpScope.Current.Directory, "ids.txt"), idInfo);
        }

        public static void WriteDebugInfo(uint fnId, uint moduleId, List<(Expr Expr, (long Min, long Max) Range)> sourceMap)
        {
            var dir = DumpScope.Current.Directory;
            var ids = ReadIds(dir);

            // stackvm id is 0
            var debugInfoDir = Path.Join(dir, "StackVMInst");
            if (!Directory.Exists(debugInfoDir))
            {
                Directory.CreateDirectory(debugInfoDir);
            }

            DumpUtility.WriteResult(
                Path.Join(dir, "StackVMInst", $"{ids[new(fnId, moduleId)]}.txt"),
                sourceMap.Where(x => x.Expr is not PrimFunctionWrapper).Select(x => ToStr(x.Expr) + x.Range).ToArray());
        }

        public static Dictionary<FunctionId, string> ReadIds(string dir)
        {
            using var sr = new StreamReader(Path.Join(dir, "ids.txt"));
            var ids = sr.ReadToEnd().Split("\n").Select(line =>
            {
                var data = line.Split(" ").ToArray();
                return (new FunctionId(uint.Parse(data[1]), uint.Parse(data[0])), data[2]);
            }).ToDictionary(x => x.Item1, x => x.Item2);
            return ids;
        }

        // todo: refactor this
        public static string ToStr(Expr expr)
        {
            string str;
            if (expr is Call call)
            {
                if (call.Target is BaseFunction fn)
                {
                    str = $"Expr: call fn_{fn.Name}";
                }
                else if (call.Target is Op o)
                {
                    str = $"Expr:{o.GetType().Name}";
                }
                else
                {
                    str = $"Expr:{expr}";
                }
            }
            else if (expr is Var v)
            {
                str = $"Expr:{v.Name}";
            }
            else if (expr is If)
            {
                str = "Expr: if";
            }
            else
            {
                str = $"Expr:{expr}";
            }

            return str + $"_{expr.GetHashCode()}";
        }

        public static void PrintAlloc(ushort localId, Expr expr, string prefix)
        {
            string? str;
            if (expr is Call call)
            {
                if (call.Target is BaseFunction fn)
                {
                    str = $"{prefix} id:{localId} Expr: call fn_{fn.Name}";
                }
                else if (call.Target is Op o)
                {
                    str = $"{prefix} id:{localId} Expr:{o.GetType().Name}";
                }
                else
                {
                    str = $"{prefix} id:{localId} Expr:{expr}";
                }
            }
            else if (expr is Var v)
            {
                str = $"{prefix} id:{localId} Expr:{v.Name}";
            }
            else if (expr is If)
            {
                str = "Expr: if";
            }
            else
            {
                str = $"{prefix} id:{localId} Expr:{expr}";
            }

            Console.WriteLine(str);
        }
    }
}
