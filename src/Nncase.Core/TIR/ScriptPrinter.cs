// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.IR;

namespace Nncase.TIR
{
    /// <summary>
    /// TIR Script printer.
    /// </summary>
    public static class ScriptPrinter
    {

        static void DumpAsScript(TextWriter textWriter, Expr expr)
        {
            var visitor = new ScriptDumpVisitor(textWriter);
            visitor.Visit(expr);
        }

        /// <summary>
        /// get this expr's il string
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public static string DumpAsScript(this Expr expr)
        {
            var builder = new StringBuilder();
            var writer = new StringWriter(builder);
            DumpAsScript(writer, expr);
            return builder.ToString();
        }

        /// <summary>
        /// dump Expr IL into `dumpDirPath/name.script`
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="name"></param>
        /// <param name="dumpDirPath"></param>
        public static void DumpAsScript(this Expr expr, string name, string dumpDirPath)
        {
            Directory.CreateDirectory(dumpDirPath);
            using var dumpFile = File.Open($"{dumpDirPath}/{name}.script", FileMode.OpenOrCreate);
            using var writer = new StreamWriter(dumpFile);
            DumpAsScript(writer, expr);
        }

        /// <summary>
        /// NOTE: 
        /// 1. each visit method create a new scope 
        /// 2. each block expr write it's string with indent!
        /// 3. each leaf expr eg. const/var write without indent!
        /// 4. each method write final without newline!
        /// </summary>
        private class ScriptDumpVisitor : ExprFunctor<StringBuilder, StringBuilder>
        {
            private readonly IRPrinter.ScopeWriter Scope;

            readonly Dictionary<Expr, StringBuilder> Docs = new(new RecordRefComparer<Expr>());

            public ScriptDumpVisitor(TextWriter textWriter)
            {
                Scope = new(textWriter);
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Call expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                var target = Visit(expr.Target);
                var args = expr.Parameters.Select(Visit).ToArray();
                Scope.Push();
                switch (expr.Target)
                {
                    case Binary binary:
                        Scope.Append($"({args[0]} {target} {args[1]})");
                        break;
                    case TIR.Store store:
                        Scope.Append($"{args[0]}[{args[1]}] = {args[2]}");
                        break;
                    case TIR.Load load:
                        Scope.Append($"{args[0]}[{args[1]}]");
                        break;
                    default:
                        Scope.Append($"{target}({string.Join<StringBuilder>(", ", args)})");
                        break;
                };
                doc = Scope.Pop();
                Docs.Add(expr, doc);
                return doc;
            }
            /// <inheritdoc/>
            public override StringBuilder Visit(Const expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                if (expr.ValueType is TensorType ttype && ttype.IsScalar)
                {
                    doc = new($"{expr}{(expr.ValueType is null ? "?" : VisitType(expr.ValueType))}");
                }
                else
                {
                    throw new NotSupportedException("The Tir NotSupport the Tensor Const!");
                }
                Docs.Add(expr, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Function expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                // 1. Function signature
                Scope.IndWrite($"def {expr.Name}({string.Join(", ", expr.Parameters.Select(Visit))})");
                // 2. Function body
                Scope.Append(Visit(expr.Body));
                doc = Scope.Pop();
                Docs.Add(expr, doc);
                // 3. only write all doc into root scope
                Scope.IndWrite(doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Op expr)
            {
                return new(expr switch
                {
                    Unary op => op.UnaryOp.ToString(),
                    Binary op => op.ToLiteral(),
                    _ => expr.GetType().Name,
                });
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Var expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                doc = new(expr.Name);
                Docs.Add(expr, doc);
                return doc;
            }

            /// <summary>
            /// visit loop var , we will assgin the var new name.
            /// </summary>
            /// <param name="expr"></param>
            /// <returns></returns>
            public StringBuilder VisitLoopVar(Var expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                doc = new(Scope.GetUniqueLoopVarName(expr));
                Docs.Add(expr, doc);
                return doc;
            }

            public StringBuilder VisitSymbolVar(IterVar symoblVar, Var bindVar)
            {
                if (Docs.TryGetValue(symoblVar, out var doc)) { return doc; }
                var bdoc = VisitLoopVar(bindVar);
                doc = new($"v{bdoc}");
                Docs.Add(symoblVar, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(For expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                // the for loop will not used by other expression, so we need save the whole `For` il
                Scope.Push();
                // 1. For Loop signature
                Scope.Append($"for {VisitLoopVar(expr.LoopVar)} in range({Visit(expr.Dom.Min)}, {Visit(expr.Dom.Max)})");
                if (expr.Mode != ForMode.Serial) { Scope.Append($" \"{expr.Mode}\""); }
                // 2. For Body
                Scope.Append(Visit(expr.Body));
                doc = Scope.Pop();
                Docs.Add(expr, doc);
                return doc;
            }
            /// <inheritdoc/>
            public override StringBuilder Visit(Sequential expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                Scope.AppendLine(":");
                // 1. Foreach Body
                using (Scope.IndentUp())
                {
                    foreach (var item in expr.Fields)
                    {
                        Scope.IndWriteLine(Visit(item));
                    }
                }
                doc = Scope.Pop();
                Docs.Add(expr, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Block expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                Scope.AppendLine($"with block(\"{expr.Name}\"):");
                foreach (var (iterVar, bindVar) in expr.IterVarPairs)
                {
                    string mode_doc = string.Empty;
                    switch (iterVar.Mode)
                    {
                        case IterMode.DataPar:
                            mode_doc = "range";
                            break;
                        case IterMode.CommReduce:
                            mode_doc = "reduce";
                            break;
                        case IterMode.Ordered:
                            mode_doc = "scan";
                            break;
                        case IterMode.Opaque:
                            mode_doc = "opaque";
                            break;
                        default:
                            throw new NotSupportedException($"{iterVar.Mode}");
                    }
                    Scope.IndWriteLine($"  {VisitSymbolVar(iterVar, bindVar)} = bind({Visit(iterVar.Value)}, {mode_doc}({Visit(iterVar.Dom.Min)}, {Visit(iterVar.Dom.Max)}))");
                }
                using (Scope.IndentUp())
                {
                    if (expr.InitBody.Count > 0)
                    {
                        Scope.IndWrite("with init()");
                        using (Scope.IndentUp())
                        {
                            Scope.Append(Visit(expr.InitBody));
                        }
                    }
                    foreach (var item in expr.Body)
                    {
                        Scope.IndWriteLine(Visit(item));
                    }
                }
                doc = Scope.Pop();
                Docs.Add(expr, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(BufferLoad expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                Scope.Append($"{expr.Buffer.Name}[{string.Join(", ", expr.Indices.Select(Visit))}]");
                doc = Scope.Pop();
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(BufferStore expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                Scope.Append($"{expr.Buffer.Name}[{string.Join(", ", expr.Indices.Select(Visit))}] = {Visit(expr.Value)}");
                doc = Scope.Pop();
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder VisitType(TensorType type)
            {
                if (DataTypes.IsFloat(type.DType)) return new("f");
                if (DataTypes.IsIntegral(type.DType)) return new("");
                throw new NotSupportedException($"{type.DType}");
            }

            public override StringBuilder Visit(IterVar expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                throw new InvalidOperationException("The IterVar Must Assgin Name In Visit(Block), You Need Check The IterVar Binding!");
            }
        }
    }
}
