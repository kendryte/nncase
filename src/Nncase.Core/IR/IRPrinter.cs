// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Math;

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
        public static void DumpFunctionAsIL(TextWriter textWriter, Function function)
        {
            var visitor = new ILDumpVisitor(textWriter);
            visitor.Visit(function);
        }

        public static void DumpFunctionAsIL(string dumpPath, Function function, string prefix = "")
        {
            var nprefix = prefix.Any() ? prefix + "_" : prefix;
            using var dumpFile = File.Open(Path.Combine(dumpPath, $"{nprefix}{function.Name}.il"), FileMode.Create);
            using var dumpWriter = new StreamWriter(dumpFile);
            var visitor = new ILDumpVisitor(dumpWriter);
            visitor.Visit(function);
        }

        public static void DumpExprAsIL(TextWriter textWriter, Expr expr)
        {
            var visitor = new ILDumpVisitor(textWriter);
            visitor.Visit(expr);
        }

        private class ILDumpVisitor : ExprFunctor<string, string>
        {
            private readonly TextWriter _textWriter;
            private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>();
            private int _localId = 0;
            private int _identLevel = 0;

            public ILDumpVisitor(TextWriter textWriter)
            {
                _textWriter = textWriter;
            }

            public override string Visit(Call expr)
            {
                if (_names.TryGetValue(expr, out var name))
                {
                    return name;
                }

                var target = Visit(expr.Target);
                var args = expr.Parameters.Select(Visit).ToArray();
                name = AllocateTempVar(expr);
                Ident().Write($"{name} = {target}({string.Join(", ", args)})");
                AppendCheckedType(expr.CheckedType);
                _textWriter.WriteLine();
                return name;
            }

            public override string Visit(Const expr)
            {
                if (_names.TryGetValue(expr, out var name))
                {
                    return name;
                }

                name = $"const({(expr.CheckedType is null ? string.Empty : VisitType(expr.CheckedType))})";
                _names.Add(expr, name);
                return name;
            }

            public override string Visit(Function expr)
            {
                if (_names.TryGetValue(expr, out var name))
                {
                    return name;
                }

                name = $"%{expr.Name}";
                _names.Add(expr, name);

                // 1. Function signature
                Ident().Write($"{name} = fn({string.Join(", ", expr.Parameters.Select(Visit))})");
                AppendCheckedType(expr.CheckedType);
                _textWriter.WriteLine(" {");

                // 2. Function body
                _identLevel++;
                var body = Visit(expr.Body);
                Ident().WriteLine(body);
                _identLevel--;

                // 3. Function closing
                Ident().WriteLine("}");
                return name;
            }

            public override string Visit(Op expr)
            {
                return expr switch
                {
                    Unary op => op.UnaryOp.ToString(),
                    Binary op => op.BinaryOp.ToString(),
                    _ => expr.GetType().Name,
                };
            }

            public override string Visit(Tuple expr)
            {
                if (_names.TryGetValue(expr, out var name))
                {
                    return name;
                }

                var fields = expr.Fields.Select(Visit).ToArray();
                name = AllocateTempVar(expr);
                Ident().Write($"{name} = ({string.Join(", ", fields)})");
                AppendCheckedType(expr.CheckedType);
                _textWriter.WriteLine();
                return name;
            }

            public override string Visit(Var expr)
            {
                if (_names.TryGetValue(expr, out var name))
                {
                    return name;
                }

                name = $"%{expr.Name}";
                _names.Add(expr, name);
                if (expr.CheckedType is IRType type)
                {
                    name += $": {VisitType(type)}";
                }

                return name;
            }

            public override string VisitType(AnyType type) => "any";

            public override string VisitType(CallableType type) =>
                $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

            public override string VisitType(InvalidType type) => "invalid";

            public override string VisitType(TensorType type) =>
                $"{DataTypes.GetDisplayName(type.DType)}{type.Shape}";

            public override string VisitType(TupleType type) =>
                $"({string.Join(", ", type.Fields.Select(VisitType))})";

            private string AllocateTempVar(Expr expr)
            {
                var name = $"%{_localId++}";
                _names.Add(expr, name);
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

            private void AppendCheckedType(IRType? type)
            {
                if (type is not null)
                {
                    _textWriter.Write($": {VisitType(type)}");
                }
            }
        }
    }
}
