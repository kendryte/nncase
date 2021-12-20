using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using Nncase.IR;
using Nncase.CodeGen.Compiler;

namespace Nncase.CodeGen.Builder
{
    /// <summary>
    /// convert the type/op to c name
    /// </summary>
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

    /// <summary>
    /// c runtime module impl
    /// </summary>
    public class CSourceRTModule : IRTModule
    {
        /// <summary>
        /// internal souce code
        /// </summary>
        string _sourcePath;

        Module _parentModule;
        RTFunction? _entry = null;
        public bool IsCompiled = false;
        readonly List<RTFunction> _functions = new();

        /// <summary>
        /// <see cref="CSourceRTModule"/>
        /// </summary>
        /// <param name="sourcePath"> c source code </param>
        public CSourceRTModule(Module mod, string sourcePath)
        {
            _sourcePath = sourcePath;
            _parentModule = mod;
        }

        /// <inheritdoc/>
        public string SourceText { get => File.ReadAllText(_sourcePath, Encoding.UTF8); }

        public RTFunction? Entry => _entry;

        public IReadOnlyList<RTFunction> Functions => throw new NotImplementedException();

        /// <inheritdoc/>
        public string SourceExt => "c";

        string _dllPath = "";

        public void Compile()
        {
            if (IsCompiled) { return; }
            var compiler = new Compiler.CSourceCompiler();
            _dllPath = compiler.Compile(_sourcePath);
            var dllPtr = NativeLibrary.Load(_dllPath);
            foreach (var f in _parentModule.Functions)
            {
                var funcType = f.ToDelegateType(Path.GetFileName(_dllPath));
                NativeLibrary.GetExport(dllPtr, f.Name);
                var funPtr = NativeLibrary.GetExport(dllPtr, f.Name);
                _functions.Add(new(f.Name, funPtr.BindDelegate(funcType)));
                if (f == _parentModule.Entry) { _entry = _functions.Last(); }
            }
        }

        /// <summary>
        /// invoke the module entry
        /// </summary>
        /// <param name="args">input args</param>
        /// <returns> results </returns>
        /// <exception cref="InvalidOperationException"></exception>
        public object? Invoke(params object?[]? args) => Entry?.Handle.DynamicInvoke(args) ?? throw new InvalidOperationException("This RTModule Have No Entry Function!");
    }

    /// <summary>
    /// the builder dispatcher
    /// </summary>
    public class CSourceHostBuilder : ITargetBuilder
    {
        string _sourcePath;
        public CSourceHostBuilder()
        {
            _sourcePath = CodeGenUtil.GetTempFileName("c");
        }

        /// <inheritdoc/>
        public IRTModule Build(Module mod, Target target)
        {

            using (var file = File.Open(_sourcePath, FileMode.OpenOrCreate, FileAccess.Write))
            {
                using (var writer = new StreamWriter(file))
                {
                    var visior = new CSourceHostBuildVisior(writer);
                    if (mod.Entry is null) { throw new InvalidProgramException("The Model Entry Is Null!"); }
                    if (mod.Entry.CheckedType is null && mod.Entry.InferenceType() == false) { throw new InvalidProgramException("The Model Entry Can't Inference Type!"); }
                    visior.Visit(mod.Entry);
                }
            }
            var rt = new CSourceRTModule(mod, _sourcePath);
            return rt;
        }
    }

    internal struct CSymbol
    {
        public string Type;
        public string Name;
        public CSymbol(string type, string name)
        {
            Type = type;
            Name = name;
        }

        public override string ToString() => $"{Type} {Name}";
    }

    /// <summary>
    /// visitor for the build c source code, the expr vistor return (type string , name string)
    /// </summary>
    internal class CSourceHostBuildVisior : ExprFunctor<CSymbol, string>
    {

        /// <summary>
        /// source writer .
        /// TODO we need the decl writer
        /// </summary>
        readonly IRPrinter.ScopeWriter srcScope;

        /// <summary>
        /// symbols name memo
        /// </summary>
        readonly Dictionary<Expr, CSymbol> _names = new();

        /// <summary>
        /// ssa var id count
        /// </summary>
        int _localId = 0;

        /// <summary>
        /// <see cref="CSourceHostBuildVisior"/>
        /// </summary>
        /// <param name="textWriter"></param>
        public CSourceHostBuildVisior(TextWriter textWriter)
        {
            srcScope = new IRPrinter.ScopeWriter(textWriter);
        }

        /// <inheritdoc/>
        public override CSymbol Visit(Call expr)
        {
            if (_names.TryGetValue(expr, out var symbol)) { return symbol; }
            symbol = AllocateTempVar(expr);
            _names.Add(expr, symbol);
            var target = Visit(expr.Target);
            var args = expr.Parameters.Select(Visit).ToArray();
            var type = VisitType(expr.CheckedType!);
            if (expr.Target is IR.Math.Binary bin && bin.BinaryOp is (BinaryOp.Add or BinaryOp.Sub or BinaryOp.Mul or BinaryOp.Div))
            {
                srcScope.IndWriteLine($"{symbol} = ({args[0].Name} {target.Name} {args[1].Name});");
            }
            else
            {
                srcScope.IndWriteLine($"{symbol} = {target.Name}({string.Join(", ", args.Select(x => x.Name))})");
            }
            return symbol;
        }

        /// <inheritdoc/>
        public override CSymbol Visit(Const expr)
        {
            if (_names.TryGetValue(expr, out var symbol)) { return symbol; }

            if (expr.CheckedType is TensorType ttype && ttype.IsScalar)
            {
                if (DataTypes.IsIntegral(ttype.DType))
                {
                    symbol = new("Invalid", expr.ToScalar<int>().ToString());
                }
                else if (DataTypes.IsFloat(ttype.DType))
                {
                    symbol = new("Invalid", expr.ToScalar<float>().ToString());
                };
            }
            else
            {
                throw new NotSupportedException($"Not Support {expr.CheckedType} Const");
            }
            _names.Add(expr, symbol);
            return symbol;
        }

        /// <inheritdoc/>
        public override CSymbol Visit(Function expr)
        {
            if (_names.TryGetValue(expr, out var symbol)) { return symbol; }
            symbol.Name = expr.Name;
            symbol.Type = "InvalidFunc";
            _names.Add(expr, symbol);

            // 1. Function signature
            srcScope.IndWriteLine($"{VisitType(expr.CheckedType!)} {symbol.Name}({string.Join(", ", expr.Parameters.Select(Visit))})");
            srcScope.IndWriteLine(" {");
            // 2. Function body
            using (srcScope.IndentUp())
            {
                var body = Visit(expr.Body);
                if (expr.CheckedType != TupleType.Void)
                {
                    srcScope.IndWriteLine($"return {body.Name};");
                }
            }
            // 3. Function closing
            srcScope.IndWriteLine("}");
            return symbol;
        }

        /// <inheritdoc/>
        public override CSymbol Visit(Op expr)
        {
            return expr switch
            {
                IR.Math.Binary op => new("Invalid", op.BinaryOp.toC()),
                _ => throw new NotSupportedException($"{expr.GetType().Name}")
            };
        }

        /// <inheritdoc/>
        public override CSymbol Visit(Var expr)
        {
            if (_names.TryGetValue(expr, out var symbol)) { return symbol; }
            symbol = new(VisitType(expr.CheckedType!), expr.Name);
            _names.Add(expr, symbol);
            return symbol;
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

        /// <summary>
        /// make ssa tempvar
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        private CSymbol AllocateTempVar(Call expr) => new(VisitType(expr.CheckedType!), $"tmpV_{_localId++}");
    }
}