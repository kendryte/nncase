using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using Nncase.IR;

namespace Nncase.CodeGen.Builder
{
    internal static class NameConverter
    {
        public static string toC(this BinaryOp binaryOp) => binaryOp switch
        {
            BinaryOp.Add => "+",
            BinaryOp.Sub => "-",
            BinaryOp.Mul => "*",
            BinaryOp.Div => "/",
            _ => throw new NotSupportedException($"{binaryOp}")
        };
        public static string toC(this DataType dataType) => dataType.ElemType switch
        {
            ElemType.Int8 => "int8_t",
            ElemType.Int16 => "int16_t",
            ElemType.Int32 => "int32_t",
            ElemType.Int64 => "int64_t",
            ElemType.UInt8 => "uint8_t",
            ElemType.UInt16 => "uint16_t",
            ElemType.UInt32 => "uint32_t",
            ElemType.UInt64 => "uint64_t",
            // ElemType.Float16 => "float16_t",
            ElemType.Float32 => "float",
            ElemType.Float64 => "double",
            // ElemType.BFloat16 => "bfloat16_t",
            _ => throw new NotSupportedException($"{dataType}")
        };
    }

    public class CSourceRTModule : RTModule
    {
        public CSourceRTModule(string source)
        {
            Source = source;
        }
    }

    public class CSourceBuilder : ITargetBuilder
    {
        /// <inheritdoc/>
        public RTModule Build(Module mod, Target target)
        {
            var sb = new StringBuilder();
            var visior = new CSourceBuildVisior(new StringWriter(sb));
            var rt = new CSourceRTModule(visior.Visit(mod.Entry));
            return rt;
        }
    }

    internal class CSourceBuildVisior : ExprFunctor<string, string>
    {
        private readonly TextWriter _textWriter;
        private readonly Dictionary<Expr, string> _names = new Dictionary<Expr, string>();
        private int _localId = 0;
        private int _identLevel = 0;

        public CSourceBuildVisior(TextWriter textWriter)
        {
            _textWriter = textWriter;
        }

        /// <inheritdoc/>
        public override string Visit(Call expr)
        {
            if (_names.TryGetValue(expr, out var name)) { return name; }

            var target = Visit(expr.Target);
            var args = expr.Parameters.Select(Visit).ToArray();
            name = AllocateTempVar(expr);
            var type = VisitType(expr.CheckedType);
            if (expr.Target is IR.Math.Binary bin && bin.BinaryOp is (BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div))
            {
                Ident().Write($"{name} = ({args[0]} {target} {args[1]});");
            }
            else
            {
                Ident().Write($"{name} = {target}({string.Join(", ", args)})");
            }
            _textWriter.WriteLine();
            return name;
        }

        /// <inheritdoc/>
        public override string Visit(Const expr)
        {
            if (_names.TryGetValue(expr, out var name)) { return name; }

            if (expr.CheckedType is TensorType ttype && ttype.IsScalar)
            {
                if (DataTypes.IsIntegral(ttype.DType))
                {
                    name = expr.ToScalar<int>().ToString();
                }
                else if (DataTypes.IsFloat(ttype.DType))
                {
                    name = expr.ToScalar<float>().ToString();
                };
            }
            else
            {
                throw new NotSupportedException($"Not Support {expr.CheckedType} Const");
            }
            _names.Add(expr, name);
            return name;
        }

        /// <inheritdoc/>
        public override string Visit(Function expr)
        {
            if (_names.TryGetValue(expr, out var name))
            {
                return name;
            }
            name = $"{expr.Name}";
            _names.Add(expr, name);

            // 1. Function signature
            Ident().Write($"{VisitType(expr.CheckedType)} {name}({string.Join(", ", expr.Parameters.Select(Visit))})");
            _textWriter.WriteLine(" {");
            // 2. Function body
            _identLevel++;
            var body = Visit(expr.Body);
            if (expr.CheckedType != TupleType.Void)
            {
                Ident().WriteLine($"return {body}");
            }
            _identLevel--;
            // 3. Function closing
            Ident().WriteLine("}");
            return name;
        }

        /// <inheritdoc/>
        public override string Visit(Op expr)
        {
            return expr switch
            {
                IR.Math.Binary op => op.BinaryOp.toC(),
                _ => throw new NotSupportedException($"{expr.GetType().Name}")
            };
        }

        /// <inheritdoc/>
        public override string Visit(Var expr)
        {
            if (_names.TryGetValue(expr, out var name)) { return name; }
            name = $"%{expr.Name}";
            _names.Add(expr, name);
            return name;
        }

        /// <inheritdoc/>
        public override string VisitType(CallableType type) => VisitType(type.ReturnType);

        /// <inheritdoc/>
        public override string VisitType(TensorType type)
        {
            if (!type.IsScalar && type.DType.Lanes != 1)
            {
                throw new NotSupportedException($"{type}");
            }
            return type.DType.toC();
        }

        /// <inheritdoc/>
        public override string VisitType(PointerType type)
        {
            if (type.DType.Lanes != 1)
            {
                throw new NotSupportedException($"{type}");
            }
            return $"({type.DType.toC()}*)";
        }

        /// <inheritdoc/>
        public override string VisitType(TupleType type) => type == TupleType.Void ?
          "void" :
          throw new InvalidProgramException($"The C Source Must Not Have TupleType {type}!");

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
    }
}