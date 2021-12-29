// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;
using Nncase.TIR;

namespace Nncase.IR
{
    /// <summary>
    /// IR printer.
    /// </summary>
    public static class IRPrinter
    {
        /// <summary>
        /// Dump function to IL text.
        /// </summary>
        /// <param name="textWriter">Text writer.</param>
        /// <param name="function">Function.</param>
        public static void DumpFunctionAsIL(Function function, TextWriter textWriter)
        {
            var visitor = new ILDumpVisitor(textWriter);
            visitor.Visit(function);
        }

        public static void DumpFunctionAsIL(Function function, string prefix, string dumpPath)
        {
            var nprefix = prefix.Any() ? prefix + "_" : prefix;
            Directory.CreateDirectory(dumpPath);
            using var dumpFile = File.Open(Path.Combine(dumpPath, $"{nprefix}{function.Name}.il"), FileMode.OpenOrCreate);
            using var dumpWriter = new StreamWriter(dumpFile);
            var visitor = new ILDumpVisitor(dumpWriter);
            visitor.Visit(function);
        }

        public static void DumpExprAsIL(TextWriter textWriter, Expr expr)
        {
            var visitor = new ILDumpVisitor(textWriter);
            visitor.Visit(expr);
        }

        /// <summary>
        /// get this expr's il string
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public static string DumpExprAsIL(this Expr expr)
        {
            var builder = new StringBuilder();
            var writer = new StringWriter(builder);
            DumpExprAsIL(writer, expr);
            return builder.ToString();
        }

        /// <summary>
        /// dump Expr IL into `dumpDirPath/name.il`
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="name"></param>
        /// <param name="dumpDirPath"></param>
        public static void DumpExprAsIL(this Expr expr, string name, string dumpDirPath)
        {
            Directory.CreateDirectory(dumpDirPath);
            using var dumpFile = File.Open($"{dumpDirPath}/{name}.il", FileMode.OpenOrCreate);
            using var writer = new StreamWriter(dumpFile);
            DumpExprAsIL(writer, expr);
        }

        public static string DumpTypeAsIL(this IRType type)
        {
            var builder = new StringBuilder();
            using var writer = new StringWriter(builder);
            var visitor = new ILDumpVisitor(writer);
            return visitor.VisitType(type);
        }

        /// <summary>
        /// a TextWirter, it's have Scope data struct.
        /// </summary>
        public class ScopeWriter
        {
            /// <summary>
            /// current writer.
            /// </summary>
            TextWriter Writer;

            TextWriter rootWriter;

            /// <summary>
            /// current VarNamelist
            /// </summary>
            List<string> VarNameList => VarNameStack.Peek();

            /// <summary>
            /// stack container
            /// </summary>
            readonly Stack<(StringBuilder, TextWriter)> ScopeStack = new();

            /// <summary>
            /// indent level
            /// </summary>
            public int indentLevel = 0;

            /// <summary>
            /// record the all var name's in this scope and parent's scope.
            /// </summary>
            readonly Dictionary<Expr, string> GlobalVarNameMap = new();

            /// <summary>
            /// record the all name used count.
            /// </summary>
            readonly Dictionary<string, int> GlobalNameUseMap = new();

            /// <summary>
            /// the scopes var name stack
            /// </summary>
            readonly Stack<List<string>> VarNameStack = new();

            /// <summary>
            /// ctor
            /// </summary>
            /// <param name="textWriter"></param>
            public ScopeWriter(TextWriter textWriter)
            {
                rootWriter = textWriter;
                Writer = textWriter;
                VarNameStack.Push(new());
            }

            /// <summary>
            /// push the new string writer, tempoary record the current code into this frame.
            /// </summary>
            public void Push()
            {
                StringBuilder builder = new StringBuilder();
                TextWriter writer = new StringWriter(builder);
                ScopeStack.Push((builder, writer));
                Writer = writer;

                VarNameStack.Push(new());
            }

            /// <summary>
            /// get current frame string
            /// </summary>
            /// <returns></returns>
            /// <exception cref="InvalidOperationException"></exception>
            public StringBuilder Pop()
            {
                var (builder, writer) = ScopeStack.Pop();
                writer.Dispose();
                if (ScopeStack.Count == 0)
                {
                    Writer = rootWriter;
                }
                else
                {
                    Writer = ScopeStack.Peek().Item2;
                }

                foreach (var name in VarNameStack.Pop()) { GlobalNameUseMap[name]--; }
                // VarNameList
                return builder;
            }

            /// <summary>
            /// insert indent and write
            /// </summary>
            /// <param name="value"></param>
            public void IndWrite(string? value) => Indent().Write(value);

            public void IndWrite(StringBuilder? value) => Indent().Write(value);

            /// <summary>
            /// insert indent and write line.
            /// </summary>
            /// <param name="value"></param>
            public void IndWriteLine(string? value = null) => Indent().WriteLine(value);

            public void IndWriteLine(StringBuilder? value) => Indent().WriteLine(value);

            /// <summary>
            /// Append the current line tail, without the indent.
            /// </summary>
            /// <param name="value"></param>
            public void Append(string value) => Writer.Write(value);
            public void Append(StringBuilder value) => Writer.Write(value);

            /// <summary>
            /// Append the current line tail, without the indent, but add new line
            /// </summary>
            /// <param name="value"></param>
            public void AppendLine(string value) => Writer.WriteLine(value);
            public void AppendLine(StringBuilder value) => Writer.WriteLine(value);

            /// <summary>
            /// remove last char.
            /// </summary>
            public void RemoveLast()
            {
                var sb = ScopeStack.Peek().Item1;
                sb.Remove(sb.Length - 1, 1);
            }

            /// <summary>
            /// insert the indent
            /// </summary>
            /// <returns></returns>
            private TextWriter Indent()
            {
                for (int i = 0; i < indentLevel; i++) { Writer.Write("  "); }
                return Writer;
            }
            /// <summary>
            /// add the indent level, return the indent mananger for auto indent down.
            /// </summary>
            /// <param name="indent_diff"></param>
            /// <returns></returns>
            public IndentMananger IndentUp(int indent_diff = 1)
            {
                return new(this, indent_diff);
            }

            /// <summary>
            /// mananger the wirte indent
            /// </summary>
            public class IndentMananger : IDisposable
            {
                /// <summary>
                /// the parent scope wirter
                /// </summary>
                readonly ScopeWriter Parent;
                /// <summary>
                /// the indent add/sub diff value
                /// </summary>
                readonly int indentDiff;
                /// <summary>
                /// <see cref="IndentMananger"/>
                /// </summary>
                /// <param name="parent"></param>
                /// <param name="level_diff"></param>
                public IndentMananger(ScopeWriter parent, int level_diff = 1)
                {
                    Parent = parent;
                    indentDiff = level_diff;
                    Parent.indentLevel += indentDiff;
                }

                public void Dispose()
                {
                    Parent.indentLevel -= indentDiff;
                }
            }

            /// <summary>
            /// get the unique loop var name, it allocate orderby i,j,k,l,i0,j0,k0...
            /// </summary>
            /// <param name="loopVar"></param>
            /// <param name="prefix"></param>
            /// <returns></returns>
            public string GetUniqueLoopVarName(Expr loopVar, string prefix)
            {
                if (GlobalVarNameMap.TryGetValue(loopVar, out var name))
                {
                    return name;
                }

                int TryGetDefault(string name)
                {
                    if (!GlobalNameUseMap.TryGetValue(name, out var count))
                    {
                        count = 0;
                        GlobalNameUseMap.Add(name, count);
                    }
                    return count;
                }

                var hint = (from c in new[] { "i", "j", "k", "l" }
                            let nc = prefix + c
                            let count = TryGetDefault(nc)
                            orderby count
                            select nc).First();
                var usecount = GlobalNameUseMap[hint];
                name = hint + (usecount == 0 ? string.Empty : usecount);
                GlobalNameUseMap[hint]++;
                GlobalVarNameMap.Add(loopVar, name);
                VarNameList.Add(name);
                return name;
            }
        }

        private class ILDumpVisitor : ExprFunctor<string, string>
        {
            private readonly ScopeWriter Scope;
            private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>();
            private int _localId = 0;

            public ILDumpVisitor(TextWriter textWriter)
            {
                Scope = new(textWriter);
            }
            /// <inheritdoc/>
            public override string Visit(Call expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }
                var target = Visit(expr.Target);
                var args = expr.Parameters.Select(Visit).ToArray();
                name = AllocateTempVar(expr);
                Scope.IndWrite($"{name} = {target}({string.Join(", ", args)})");
                AppendCheckedType(expr.CheckedType);
                return name;
            }
            /// <inheritdoc/>
            public override string Visit(Const expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }

                if (expr.CheckedType is TensorType ttype && ttype.IsScalar)
                {
                    name = $"const({expr} : {(expr.CheckedType is null ? string.Empty : VisitType(expr.CheckedType))})";
                }
                else
                {
                    name = $"const({(expr.CheckedType is null ? string.Empty : VisitType(expr.CheckedType))})";
                }
                _names.Add(expr, name);
                return name;
            }

            /// <inheritdoc/>
            public override string Visit(Function expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }

                name = $"%{expr.Name}";
                _names.Add(expr, name);
                Scope.Push();
                // 1. Function signature
                Scope.IndWrite($"{name} = fn({string.Join(", ", expr.Parameters.Select(Visit))})");
                AppendCheckedType(expr.CheckedType, " {\n");
                // 2. Function body
                using (Scope.IndentUp()) { var body = Visit(expr.Body); }
                // 3. Function closing
                Scope.IndWriteLine("}");
                Scope.IndWrite(Scope.Pop());
                return name;
            }
            /// <inheritdoc/>
            public override string Visit(Op expr)
            {
                return expr switch
                {
                    Unary op => op.UnaryOp.ToString(),
                    Binary op => op.BinaryOp.ToString(),
                    _ => expr.GetType().Name,
                };
            }
            /// <inheritdoc/>
            public override string Visit(Tuple expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }
                var fields = expr.Fields.Select(Visit).ToArray();
                name = AllocateTempVar(expr);
                Scope.IndWrite($"{name} = ({string.Join(", ", fields)})");
                AppendCheckedType(expr.CheckedType);
                Scope.IndWriteLine();
                return name;
            }
            /// <inheritdoc/>
            public override string Visit(Var expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }
                name = $"%{expr.Name}";
                _names.Add(expr, name);
                if (expr.CheckedType is IRType type) { name += $": {VisitType(type)}"; }
                return name;
            }
            /// <inheritdoc/>
            public override string Visit(For expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }
                // the for loop will not used by other expression, so we need save the whole `For` il
                Scope.Push();
                // 1. For Loop signature
                Scope.Append($"For {expr.Mode}({Visit(expr.LoopVar)} in Range({Visit(expr.Dom.Min)}, {Visit(expr.Dom.Max)})");
                AppendCheckedType(expr.CheckedType, " {\n");
                // 2. For Body
                using (Scope.IndentUp())
                {
                    Visit(expr.Body!);
                }
                // 3. For closing
                Scope.IndWriteLine("}");

                // 4. extact whole il
                Scope.IndWrite(Scope.Pop());
                return "";
            }
            /// <inheritdoc/>
            public override string Visit(Sequential expr)
            {
                if (_names.TryGetValue(expr, out var name)) { return name; }
                Scope.Push();
                // 1. Sequential signature
                Scope.Append($"Sequential");
                AppendCheckedType(expr.CheckedType, " {\n");
                // 2. For Body
                using (Scope.IndentUp())
                {
                    foreach (var item in expr.Fields) { Visit(item); }
                }
                // 3. For closing
                Scope.IndWriteLine("}");

                // 4. extact whole il
                Scope.IndWrite(Scope.Pop());
                return "";
            }

            /// <inheritdoc/>
            public override string VisitType(AnyType type) => "any";

            /// <inheritdoc/>
            public override string VisitType(CallableType type) =>
                $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";
            /// <inheritdoc/>
            public override string VisitType(InvalidType type) => $"invalid:{type.Reason}";
            /// <inheritdoc/>
            public override string VisitType(TensorType type) =>
                $"{DataTypes.GetDisplayName(type.DType)}{type.Shape}";

            /// <inheritdoc/>
            public override string VisitType(TupleType type) =>
                $"({string.Join(", ", type.Fields.Select(VisitType))})";

            /// <inheritdoc/>
            public override string VisitType(HandleType type)
            {
                return $"pointer:{DataTypes.GetDisplayName(type.DType)}";
            }

            private string AllocateTempVar(Expr expr)
            {
                var name = $"%{_localId++}";
                _names.Add(expr, name);
                return name;
            }
            private void AppendCheckedType(IRType? type, string end = "\n")
            {
                if (type is not null)
                {
                    Scope.Append($": {VisitType(type)}{end}");
                }
            }
        }
    }
}
