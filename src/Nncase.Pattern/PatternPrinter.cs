// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Pattern
{
    /// <summary>
    /// IR printer.
    /// </summary>
    public static class PatternPrinter
    {
        /// <summary>
        /// dump pattern to writer.
        /// </summary>
        /// <param name="textWriter"></param>
        /// <param name="pattern"></param>
        public static void DumpAsIL(TextWriter textWriter, ExprPattern pattern)
        {
            var visitor = new ILDumpVisitor(textWriter);
            visitor.Visit(pattern);
        }

        /// <summary>
        /// dump pattern to string.
        /// </summary>
        /// <param name="pattern"></param>
        /// <returns></returns>
        public static string DumpAsIL(this ExprPattern pattern)
        {
            var builder = new StringBuilder();
            var writer = new StringWriter(builder);
            DumpAsIL(writer, pattern);
            return builder.ToString();
        }

        /// <summary>
        /// dump the pattern to file.
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="name"></param>
        /// <param name="dumpPath"></param>
        public static void DumpAsIL(this ExprPattern pattern, string name, string dumpPath)
        {
            Directory.CreateDirectory(dumpPath);
            using var dumpFile = File.Open($"{dumpPath}/{name}.il", FileMode.OpenOrCreate);
            using var writer = new StreamWriter(dumpFile);
            DumpAsIL(writer, pattern);
        }

        private class ILDumpVisitor : PatternFunctor<string, string>
        {
            private readonly TextWriter _textWriter;
            private readonly Dictionary<ExprPattern, string> _names = new();
            private int _localId = 0;
            private int _identLevel = 0;

            public ILDumpVisitor(TextWriter textWriter)
            {
                _textWriter = textWriter;
            }

            /// <inheritdoc/>
            public override string Visit(CallPattern pattern)
            {
                if (_names.TryGetValue(pattern, out var name))
                {
                    return name;
                }

                var target = Visit(pattern.Target);
                var args = pattern.Parameters.Select(Visit).ToArray();
                name = AllocateTempVar(pattern);
                Ident().Write($"{name} = {target}({string.Join(", ", args)})");
                AppendType(pattern.CheckedTypePat);
                _textWriter.WriteLine();
                return name;
            }

            /// <inheritdoc/>
            public override string Visit(ConstPattern pattern)
            {
                if (_names.TryGetValue(pattern, out var name))
                {
                    return name;
                }

                if (pattern.Target is not null)
                {
                    name = $"const({pattern.Target})";
                }
                else
                {
                    name = $"const({VisitType(pattern.CheckedTypePat)})";
                }

                _names.Add(pattern, name);
                return name;
            }

            /// <inheritdoc/>
            public override string Visit(FunctionPattern pattern)
            {
                if (_names.TryGetValue(pattern, out var name))
                {
                    return name;
                }

                name = $"%func";
                _names.Add(pattern, name);

                // 1. Function signature
                Ident().Write($"{name} = fn({string.Join(", ", pattern.Parameters.Select(Visit))})");

                // AppendType(pattern.CheckedTypePat);
                _textWriter.WriteLine(" {");

                // 2. Function body
                _identLevel++;
                var body = Visit(pattern.Body);
                Ident().WriteLine(body);
                _identLevel--;

                // 3. Function closing
                Ident().WriteLine("}");
                return name;
            }

            /// <inheritdoc/>
            public override string Visit(OpPattern pattern)
            {
                return pattern switch
                {
                    // todo we need parser the cond func to string.
                    _ => pattern.GetType().Name,
                };
            }

            /// <inheritdoc/>
            public override string Visit(TuplePattern expr)
            {
                if (_names.TryGetValue(expr, out var name))
                {
                    return name;
                }

                var fields = expr.Fields.Select(Visit).ToArray();
                name = AllocateTempVar(expr);
                Ident().Write($"{name} = ({string.Join(", ", fields)})");
                _textWriter.WriteLine();
                return name;
            }

            /// <inheritdoc/>
            public override string Visit(VarPattern pattern)
            {
                if (_names.TryGetValue(pattern, out var name))
                {
                    return name;
                }

                name = $"%{pattern.Name}";
                _names.Add(pattern, name);
                name += $": {VisitType(pattern.CheckedTypePat)}";
                return name;
            }

            /// <inheritdoc/>
            public override string Visit(WildCardPattern pattern)
            {
                if (_names.TryGetValue(pattern, out var name))
                {
                    return name;
                }

                name = $"%{pattern.Name}";
                _names.Add(pattern, name);
                name += $": {VisitType(pattern.CheckedTypePat)}";
                return name;
            }

            /// <inheritdoc/>
            public override string VisitType(TypePattern pattern) => pattern.Reason;

            private string AllocateTempVar(ExprPattern pattern)
            {
                var name = $"%{_localId++}";
                _names.Add(pattern, name);
                return name;
            }

            private TextWriter Ident()
            {
                for (int i = 0; i < _identLevel; i++)
                {
                    _textWriter.Write("    ");
                }

                return _textWriter;
            }

            private void AppendType(TypePattern? type)
            {
                if (type is not null)
                {
                    _textWriter.Write($": {VisitType(type)}");
                }
            }
        }
    }
}
