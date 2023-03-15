// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern printer.
/// </summary>
public static class PatternPrinter
{
    /// <summary>
    /// dump pattern to writer.
    /// </summary>
    /// <param name="textWriter">Text writer.</param>
    /// <param name="pattern">Pattern.</param>
    public static void DumpAsIL(TextWriter textWriter, IPattern pattern)
    {
        var visitor = new ILDumpVisitor(textWriter);
        visitor.Visit(pattern);
    }

    /// <summary>
    /// dump pattern to string.
    /// </summary>
    /// <param name="pattern">Pattern.</param>
    /// <returns>Dumped il.</returns>
    public static string DumpAsIL(this IPattern pattern)
    {
        var builder = new StringBuilder();
        var writer = new StringWriter(builder);
        DumpAsIL(writer, pattern);
        return builder.ToString();
    }

    /// <summary>
    /// dump the pattern to file.
    /// </summary>
    public static void DumpAsIL(this IPattern pattern, string name, string dumpPath)
    {
        Directory.CreateDirectory(dumpPath);
        using var dumpFile = File.Open($"{dumpPath}/{name}.il", FileMode.OpenOrCreate);
        using var writer = new StreamWriter(dumpFile);
        DumpAsIL(writer, pattern);
    }

    private class ILDumpVisitor : PatternFunctor<string, string>
    {
        private readonly TextWriter _textWriter;
        private readonly Dictionary<IPattern, string> _names = new();
        private int _localId;
        private int _identLevel;

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
            var args = pattern.Arguments.Select(Visit).ToArray();
            name = AllocateTempVar(pattern);
            Ident().Write($"{name} = {target}({string.Join(", ", args)})");
            AppendType(pattern.TypePattern);
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

            if (pattern.Value is not null)
            {
                name = $"const({pattern.Value})";
            }
            else
            {
                name = $"const({VisitType(pattern.TypePattern)})";
            }

            _names.Add(pattern, name);
            return name;
        }

        /// <inheritdoc/>
        public override string Visit(TensorConstPattern pattern)
        {
            if (_names.TryGetValue(pattern, out var name))
            {
                return name;
            }

            if (pattern.Value is not null)
            {
                name = $"const({pattern.Value})";
            }
            else
            {
                name = $"const({VisitType(pattern.TypePattern)})";
            }

            _names.Add(pattern, name);
            return name;
        }

        /// <inheritdoc/>
        public override string Visit(TupleConstPattern pattern)
        {
            if (_names.TryGetValue(pattern, out var name))
            {
                return name;
            }

            if (pattern.Value is not null)
            {
                name = $"const({pattern.Value})";
            }
            else
            {
                name = $"const({VisitType(pattern.TypePattern)})";
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
        public override string Visit(IOpPattern pattern)
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

            name = "var";
            _names.Add(pattern, name);
            name += $": {VisitType(pattern.TypePattern)}";
            return name;
        }

        /// <inheritdoc/>
        public override string Visit(ExprPattern pattern)
        {
            if (_names.TryGetValue(pattern, out var name))
            {
                return name;
            }

            name = pattern.IsWildcard ? "?" : "expr";
            _names.Add(pattern, name);
            name += $": {VisitType(pattern.TypePattern)}";
            return name;
        }

        /// <inheritdoc/>
        public override string VisitType(TypePattern? pattern) => pattern?.Reason ?? string.Empty;

        private string AllocateTempVar(IPattern pattern)
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
