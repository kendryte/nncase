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

namespace Nncase.TIR;

/// <summary>
/// TIR Script printer.
/// </summary>
public static class ScriptPrinter
{
    /// <summary>
    /// dump Expr as c# script.
    /// </summary>
    /// <param name="prim_func">the input function.</param>
    /// <param name="textWriter">writer.</param>
    public static void DumpAsScript(PrimFunction prim_func, TextWriter textWriter)
    {
        var visitor = new ScriptDumpVisitor(textWriter);
        visitor.Visit(prim_func);
    }

    private record ScriptSymobl(StringBuilder FullSpan, string Name, bool IsRefSymobl)
    {
        int PrintCount = 0;
        public ScriptSymobl(StringBuilder FullSpan) : this(FullSpan, string.Empty, false) { }

        public override string? ToString()
        {
            if (IsRefSymobl && PrintCount > 0)
                return Name;
            if (IsRefSymobl && PrintCount == 0)
                PrintCount++;
            return FullSpan.ToString();
        }
    }

    /// <summary>
    /// NOTE:
    /// 1. each visit method create a new scope
    /// 2. each block expr's start with newline and indent
    ///
    /// <example>
    /// `indent` if (x){
    /// `indent` &lt;- the current block start from here.
    /// `indent` }&lt;- end without new line.
    /// </example>
    ///
    /// 3. each block expr's end without newline
    /// <example>
    /// `indent` if (x){
    /// `indent` `indent` x++;
    /// `indent` }&lt;- end without new line.
    /// </example>
    ///
    /// 4. in block expr, each line expr like const/var write without indent!.
    /// </summary>
    private class ScriptDumpVisitor : ExprFunctor<ScriptSymobl, string>
    {
        private readonly IRPrinter.ScopeWriter Scope;

        readonly Dictionary<Expr, ScriptSymobl> Context = new(ReferenceEqualityComparer.Instance);

        public ScriptDumpVisitor(TextWriter textWriter)
        {
            Scope = new(textWriter);
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Call expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
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
                case Function:
                case TIR.PrimFunction:
                    Scope.AppendLine("");
                    Scope.IndWrite($"{target.Name}({string.Join(", ", (from a in args select a.ToString()))})");
                    break;
                default:
                    Scope.Append($"{target}({string.Join(", ", (from a in args select a.ToString()))})");
                    break;
            }
            doc = new(Scope.Pop());
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Const expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            if (expr.ValueType is TensorType ttype && ttype.IsScalar)
            {
                doc = new(new($"{expr}"));
            }
            else
            {
                throw new NotSupportedException("The Tir NotSupport the Tensor Const!");
            }

            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(PrimFunction expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();

            // 1. Function signature

            Scope.IndWrite($"T.PrimFunc(\"{expr.Name}\", {string.Join(", ", expr.Parameters.Select(Visit))}).Body(");
            Scope.Append(" // " + VisitType(expr.CheckedType!));

            // 2. Function body
            Scope.Append(Visit(expr.Body).FullSpan);
            Scope.IndWrite(");");
            doc = new(Scope.Pop(), expr.Name, true);
            Context.Add(expr, doc);

            // 3. only write all doc into root scope
            Scope.Append(doc.FullSpan);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Op expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            doc = new(new(expr switch
            {
                Unary op => op.UnaryOp.ToString(),
                Binary op => op.ToLiteral(),
                IR.Tensors.Cast op => "Cast",
                _ => expr.GetType().Name,
            }));
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Var expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            doc = new(new(expr.Name));
            Context.Add(expr, doc);
            return doc;
        }

        /// <summary>
        /// visit loop var , we will assgin the var new name.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="prefix"> the prefix for this var name.</param>
        /// <returns></returns>
        public ScriptSymobl VisitLoopVar(Expr expr, string prefix = "")
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            doc = new(new(Scope.GetUniqueLoopVarName(expr, prefix)));
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(For expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }

            // the for loop will not used by other expression, so we need save the whole `For` il
            Scope.Push();

            // 1. For Loop signature
            var i_name = VisitLoopVar(expr.LoopVar);
            Scope.Append($"T.{expr.Mode}(out var {i_name}, ({Visit(expr.Dom.Start)}, {Visit(expr.Dom.Stop)}, {Visit(expr.Dom.Step)}), out var f{i_name}).Body(");
            Scope.Append(" // " + VisitType(expr.CheckedType!));

            // 2. For Body
            Scope.Append(Visit(expr.Body).FullSpan);
            Scope.IndWrite(")");
            doc = new(Scope.Pop());
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Sequential expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();
            Scope.AppendLine("");

            // 1. Foreach Body
            using (Scope.IndentUp())
            {
                foreach (var item in expr.Fields)
                {
                    Scope.IndWriteLine(Visit(item).FullSpan);
                }
            }

            doc = new(Scope.Pop());
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Block expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
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
                Scope.IndWriteLine($"Bind(out var {Visit(iterVar)}, ({Visit(iterVar.Dom.Start)}, {Visit(iterVar.Dom.Stop)}, ({Visit(iterVar.Dom.Step)})), IterMode.{iterVar.Mode}, {Visit(iterVar.Value)}).");
            }

            // 3. write init body
            if (expr.InitBody.Count > 0)
            {
                Scope.IndWriteLine("Init(");
                foreach (var item in expr.InitBody)
                {
                    Scope.IndWriteLine(Visit(item).FullSpan);
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
                foreach (var item in expr.Body)
                {
                    Scope.IndWriteLine(Visit(item).FullSpan);
                }
            }

            Scope.IndWrite(")");
            doc = new(Scope.Pop());
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(BufferLoad expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();
            Scope.Append($"{expr.Buffer.Name}[{string.Join(", ", expr.Indices.Select(Visit))}]");
            doc = new(Scope.Pop());
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(BufferStore expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();
            Scope.Append($"{expr.Buffer.Name}[{string.Join(", ", expr.Indices.Select(Visit))}] = {Visit(expr.Value)}");
            doc = new(Scope.Pop());
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(IterVar expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            return VisitLoopVar(expr, "v");
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(IfThenElse expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();
            Scope.Append($"T.If({Visit(expr.Condition)}).Then(");
            Scope.AppendLine($" // {VisitType(expr.CheckedType!)}");
            using (Scope.IndentUp())
            {
                foreach (var item in (Sequential)expr.Then)
                {
                    Scope.IndWriteLine(Visit(item).FullSpan);
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
                        Scope.IndWriteLine(Visit(item).FullSpan);
                    }
                }

                Scope.IndWrite(")");
            }

            doc = new(Scope.Pop());
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override ScriptSymobl Visit(Let expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();
            Scope.Append($"T.Let({Visit(expr.Var)}, {Visit(expr.Expression)}).Body(");
            Scope.AppendLine($" // {VisitType(expr.CheckedType!)}");
            using (Scope.IndentUp(1))
            {
                foreach (var item in (Sequential)expr.Body)
                {
                    Scope.IndWriteLine(Visit(item).FullSpan);
                }
            }
            Scope.IndWrite(")");
            doc = new(Scope.Pop());
            Context.Add(expr, doc);
            return doc;
        }

        public override ScriptSymobl Visit(Buffer expr)
        {
            if (Context.TryGetValue(expr, out var doc)) { return doc; }
            Scope.Push();
            Scope.Append($"T.MemRef({expr.Name}, {VisitType(expr.ElemType)})");
            doc = new(Scope.Pop(), expr.Name, true);
            Context.Add(expr, doc);
            return doc;
        }

        /// <inheritdoc/>
        public override string VisitType(TensorType type) => type.DType switch
        {
            PrimType ptype => $"{ptype.GetDisplayName()}{type.Shape}",
            PointerType { ElemType: PrimType etype } ptype => $"*{etype.GetDisplayName()}",
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
