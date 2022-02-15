// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Math;

namespace Nncase.TIR
{
    /// <summary>
    /// TIR Script printer.
    /// </summary>
    public static class ScriptPrinter
    {
        /// <summary>
        /// dump the ir as t script
        /// </summary>
        /// <param name="textWriter"></param>
        /// <param name="expr"></param>
        static void DumpAsScript(TextWriter textWriter, Expr expr)
        {
            var visitor = new ScriptDumpVisitor(textWriter);
            visitor.Visit(expr);
        }

        /// <summary>
        /// get this expr's il string.
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
        /// dump Expr IL into `dumpDirPath/name.script`.
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
        /// 2. each block expr's start with newline and indent
        ///
        /// <example>
        /// `indent` if (x){
        /// `indent` <- the current block start from here.
        /// `indent` }<- end without new line.
        /// </example>
        ///
        /// 3. each block expr's end without newline
        /// <example>
        /// `indent` if (x){
        /// `indent` `indent` x++;
        /// `indent` }<- end without new line.
        /// </example>
        ///
        /// 4. in block expr, each line expr like const/var write without indent!.
        /// </summary>
        private class ScriptDumpVisitor : ExprFunctor<StringBuilder, string>
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
                    case IR.Tensors.Cast cast:
                        Scope.Append($"{target}({args[0]}, {cast.NewType})");
                        break;
                    default:
                        Scope.Append($"{target}({string.Join<StringBuilder>(", ", args)})");
                        break;
                }

                ;
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
                    doc = new($"{expr}");
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

                Scope.IndWrite($"T.PrimFunc(\"{expr.Name}\", {string.Join(", ", expr.Parameters.Select(Visit))}).Body(");
                Scope.Append(" // " + VisitType(expr.CheckedType!));

                // 2. Function body
                Scope.Append(Visit(expr.Body));
                Scope.IndWrite(");");
                doc = Scope.Pop();
                Docs.Add(expr, doc);

                // 3. only write all doc into root scope
                Scope.IndWrite(doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Op expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                doc = new(expr switch
                {
                    Unary op => op.UnaryOp.ToString(),
                    Binary op => op.ToLiteral(),
                    IR.Tensors.Cast op => "Cast",
                    _ => expr.GetType().Name,
                });
                Docs.Add(expr, doc);
                return doc;
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
            /// <param name="prefix"> the prefix for this var name.</param>
            /// <returns></returns>
            public StringBuilder VisitLoopVar(Expr expr, string prefix = "")
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                doc = new(Scope.GetUniqueLoopVarName(expr, prefix));
                Docs.Add(expr, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(For expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }

                // the for loop will not used by other expression, so we need save the whole `For` il
                Scope.Push();

                // 1. For Loop signature
                var i_name = VisitLoopVar(expr.LoopVar);
                Scope.Append($"T.{expr.Mode}(out var {i_name}, ({Visit(expr.Dom.Min)}, {Visit(expr.Dom.Max)}), out var f{i_name}).Body(");
                Scope.Append(" // " + VisitType(expr.CheckedType!));

                // 2. For Body
                Scope.Append(Visit(expr.Sequence));
                Scope.IndWrite(")");
                doc = Scope.Pop();
                Docs.Add(expr, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(Sequential expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                Scope.AppendLine("");

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

                // 1. write head
                Scope.AppendLine($"T.Block(\"{expr.Name}\").");

                // 2. write iter var bind
                foreach (var iterVar in expr.IterVars)
                {
                    string mode_doc = string.Empty;
                    switch (iterVar.Mode)
                    {
                        case IterationMode.DataParallel:
                            mode_doc = "S";
                            break;
                        case IterationMode.CommReduce:
                            mode_doc = "R";
                            break;
                        case IterationMode.Ordered:
                            mode_doc = "scan";
                            break;
                        case IterationMode.Opaque:
                            mode_doc = "opaque";
                            break;
                        default:
                            throw new NotSupportedException($"{iterVar.Mode}");
                    }

                    // Scope.IndWriteLine($"Remap(out var {VisitSymbolVar(iterVar, loop.LoopVar)}, f{VisitLoopVar(loop.LoopVar)}, \'{mode_doc}\').");
                    Scope.IndWriteLine($"Bind(out var {Visit(iterVar)}, ({Visit(iterVar.Dom.Min)}, {Visit(iterVar.Dom.Max)}), IterMode.{iterVar.Mode}, {Visit(iterVar.Value)}).");
                }

                // 3. write init body
                if (expr.InitSequence.Count > 0)
                {
                    Scope.IndWriteLine("Init(");
                    foreach (var item in expr.InitSequence)
                    {
                        Scope.IndWriteLine(Visit(item));
                    }

                    Scope.IndWrite(").");
                }
                else
                {
                    Scope.RemoveLast();
                }

                // 4. wirte body
                Scope.Append("Body(");
                Scope.AppendLine(" // " + VisitType(expr.CheckedType!));
                using (Scope.IndentUp())
                {
                    foreach (var item in expr.Sequence)
                    {
                        Scope.IndWriteLine(Visit(item));
                    }
                }

                Scope.IndWrite(")");
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
            public override StringBuilder Visit(IterVar expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                return VisitLoopVar(expr, "v");
            }

            /// <inheritdoc/>
            public override StringBuilder Visit(IfThenElse expr)
            {
                if (Docs.TryGetValue(expr, out var doc)) { return doc; }
                Scope.Push();
                Scope.Append($"T.If({Visit(expr.Condition)}).Then(");
                Scope.AppendLine($" // {VisitType(expr.CheckedType!)}");
                using (Scope.IndentUp())
                {
                    foreach (var item in (Sequential)expr.Then)
                    {
                        Scope.IndWriteLine(Visit(item));
                    }
                }

                Scope.IndWrite(")");
                if (((Sequential)expr.Else).Count > 0)
                {
                    Scope.AppendLine(".Then(");
                    using (Scope.IndentUp())
                    {
                        foreach (var item in (Sequential)expr.Else)
                        {
                            Scope.IndWriteLine(Visit(item));
                        }
                    }

                    Scope.IndWrite(")");
                }

                doc = Scope.Pop();
                Docs.Add(expr, doc);
                return doc;
            }

            /// <inheritdoc/>
            public override string VisitType(TensorType type) => type.DType switch
            {
                PrimType ptype => $"{ptype}{type.Shape}",
                PointerType { ElemType: PrimType etype } ptype => $"Handle:{etype}",
                _ => throw new NotSupportedException(type.DType.GetType().Name),
            };

            /// <inheritdoc/>
            public override string VisitType(CallableType type) =>
                $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

            /// <inheritdoc/>
            public override string VisitType(TupleType type) =>
                $"({string.Join(", ", type.Fields.Select(VisitType))})";

            /// <inheritdoc/>
            public override string VisitType(InvalidType type) => $"Invalid:{type.Reason}";
        }
    }
}
