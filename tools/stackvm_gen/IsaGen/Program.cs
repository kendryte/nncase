// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using RazorLight;
using RazorLight.Razor;

namespace IsaGen;

public class IsaExtractor
{
    private readonly RazorLightEngine _engine;
    private readonly List<Instruction> _insts;
    private readonly List<Type> _tensorInsts;
    private readonly List<Type> _enums;
    private readonly HashSet<Type> _generatedEnums = new HashSet<Type>();

    public IsaExtractor()
    {
        _engine = new RazorLightEngineBuilder()
            .UseMemoryCachingProvider()
            .UseProject(new EmbeddedRazorProject(typeof(Program)) { Extension = ".razor" })
            .Build();

        _insts = (from t in typeof(Instruction).Assembly.ExportedTypes
                  where !t.IsAbstract && t.IsAssignableTo(typeof(Instruction))
                  orderby t.MetadataToken
                  select (Instruction)Activator.CreateInstance(t)!).ToList();
        _enums = (from t in typeof(Instruction).Assembly.ExportedTypes
                  where t.IsEnum && t.IsDefined(typeof(EnumNameAttribute))
                  orderby t.MetadataToken
                  select t).ToList();
        _tensorInsts = (from t in typeof(Nncase.CoreApplicationPart).Assembly.ExportedTypes
                        where !t.IsAbstract && t.IsAssignableTo(typeof(Nncase.IR.Op))
                        where t.Namespace!.StartsWith("Nncase.IR.")
                        where t.Namespace != "Nncase.IR.Buffers"
                        select t).OrderBy(x => x.Name).ToList();
    }

    public IReadOnlyList<KeyValuePair<string, IReadOnlyList<InstructionInfo>>> Instructions { get; private set; } = Array.Empty<KeyValuePair<string, IReadOnlyList<InstructionInfo>>>();

    public IReadOnlyList<KeyValuePair<string, IReadOnlyList<InstructionInfo>>> TensorInstructions { get; private set; } = Array.Empty<KeyValuePair<string, IReadOnlyList<InstructionInfo>>>();

    public List<EnumInfo> Enums { get; private set; } = new();

    public static string ToBinaryString(uint value, uint bits)
    {
        return Convert.ToString(value, 2).PadLeft((int)bits, '0');
    }

    public void Extract()
    {
        Instructions = (from inst in _insts
                        let t = inst.GetType()
                        let c = t.GetCustomAttribute<CategoryAttribute>()!.Category
                        let fs = GetInstructionFields(inst, t)
                        group new InstructionInfo(
                            Index: (int)inst.OpCode,
                            CppName: t.GetCustomAttribute<DisplayNameAttribute>()!.DisplayName,
                            CSharpName: t.Name.Replace("Instruction", string.Empty, StringComparison.Ordinal),
                            Category: c,
                            OpCode: inst.OpCode,
                            Description: t.GetCustomAttribute<DescriptionAttribute>()!.Description,
                            Fields: fs,
                            Inputs: new()) by c).Select(x => new KeyValuePair<string, IReadOnlyList<InstructionInfo>>(x.Key, x.ToList())).ToList();

        Enums = (from e in _enums
                 let b = e.GetCustomAttribute<BrowsableAttribute>()
                 where b == null || b.Browsable
                 select new EnumInfo(
                     CppName: e.GetCustomAttribute<EnumNameAttribute>()!.Name,
                     UnderlyingCppType: CppFieldType(e.GetEnumUnderlyingType()),
                     UnderlyingCSharpType: CSharpFieldType(e.GetEnumUnderlyingType()),
                     Length: FieldLength(e),
                     Fields: GetEnumFields(e))).ToList();

        TensorInstructions = (from t in _tensorInsts.Select((x, i) => (x, i))
                              let c = t.x.Namespace!.Replace("Nncase.IR.", string.Empty, StringComparison.Ordinal)
                              let fs = GetTensorInstructionFields(t.i, t.x)
                              let inputs = GetTensorInstructionInputs(t.x)
                              group new InstructionInfo(
                                  Index: t.i,
                                  CppName: SnakeName(t.x.Name),
                                  CSharpName: PascalName(t.x.Name),
                                  Category: c,
                                  OpCode: OpCode.TENSOR,
                                  Description: string.Empty,
                                  Fields: fs,
                                  Inputs: inputs) by c).Select(x => new KeyValuePair<string, IReadOnlyList<InstructionInfo>>(x.Key, x.ToList())).ToList();

        AddTensorFunctionEnum();
    }

    public Task<string> RenderAsync(string templateName)
    {
        return _engine.CompileRenderAsync(templateName, this);
    }

    private void AddTensorFunctionEnum()
    {
        var fields = (from t in TensorInstructions.SelectMany(x => x.Value)
                      select new EnumFieldInfo(
                          CppName: t.CppName,
                          Value: (uint)t.Index,
                          Description: string.Empty)).ToList();
        var e = new EnumInfo(
                CppName: "tensor_function_t",
                UnderlyingCppType: CppFieldType(typeof(ushort)),
                UnderlyingCSharpType: CSharpFieldType(typeof(ushort)),
                Length: FieldLength(typeof(ushort)),
                Fields: fields);
        Enums.Insert(Enums.FindIndex(x => x.CppName == "opcode_t") + 1, e);
    }

    private List<EnumFieldInfo> GetEnumFields(Type e)
    {
        return (from f in e.GetFields(BindingFlags.Public | BindingFlags.Static)
                select new EnumFieldInfo(
                    CppName: e == typeof(OpCode) ? f.Name : SnakeName(f.Name),
                    Value: (uint)Convert.ToInt32(f.GetValue(null)),
                    Description: f.GetCustomAttribute<DescriptionAttribute>()?.Description ?? string.Empty)).ToList();
    }

    private string SnakeName(string name)
    {
        if (name == "OpCode")
        {
            return "opcode";
        }

        var sb = new StringBuilder();
        bool lastCapital = true;
        bool lastIsLetter = true;
        foreach (var c in name)
        {
            var isLetter = char.IsLetter(c);
            var isCaptial = isLetter ? char.IsUpper(c) : false;
            if (!lastCapital && isCaptial && sb.Length != 0)
            {
                if (lastIsLetter || c != 'D')
                {
                    sb.Append('_');
                }
            }

            sb.Append(char.ToLowerInvariant(c));

            if (!lastIsLetter && c == 'D')
            {
                sb.Append('_');
            }

            lastCapital = isCaptial;
            lastIsLetter = isLetter;
        }

        return sb.ToString().Trim('_');
    }

    private string SnakeTypeName(string name)
    {
        return SnakeName(name) + "_t";
    }

    private string PascalName(string name)
    {
        return name;
    }

    private string CamelName(string name)
    {
        return char.ToLowerInvariant(name[0]) + name.Substring(1);
    }

    private void GenerateEnum(Type type)
    {
        if (_generatedEnums.Contains(type))
        {
            return;
        }

        _generatedEnums.Add(type);
        Enums.Add(new EnumInfo(
                CppName: SnakeTypeName(type.Name),
                UnderlyingCppType: CppFieldType(type.GetEnumUnderlyingType()),
                UnderlyingCSharpType: CSharpFieldType(type.GetEnumUnderlyingType()),
                Length: FieldLength(type),
                Fields: GetEnumFields(type)));
    }

    private List<InstructionField> GetInstructionFields(Instruction inst, Type t)
    {
        var props = new List<(int, PropertyInfo)>();
        var fields = new List<InstructionField>();
        foreach (var f in t.GetProperties())
        {
            int metadataToken = f.MetadataToken;
            var baseType = t.BaseType;
            while (baseType != typeof(Instruction)
                && baseType != null)
            {
                var bf = baseType.GetProperty(f.Name);
                if (bf != null)
                {
                    metadataToken = bf.MetadataToken;
                    baseType = baseType.BaseType;
                }
                else
                {
                    break;
                }
            }

            props.Add((metadataToken, f));
        }

        props.Sort((a, b) => a.Item1 - b.Item1);

        foreach (var (m, f) in props)
        {
            var len = FieldLength(f.PropertyType);
            fields.Add(new InstructionField(
                            CppName: f.GetCustomAttribute<DisplayNameAttribute>()!.DisplayName,
                            CSharpName: f.GetCustomAttribute<DisplayNameAttribute>()!.DisplayName,
                            CSharpPropName: f.GetCustomAttribute<DisplayNameAttribute>()!.DisplayName,
                            CppType: CppFieldType(f.PropertyType),
                            CSharpType: CSharpFieldType(f.PropertyType),
                            UnderlyingCSharpType: f.PropertyType.IsEnum ? CSharpFieldType(f.PropertyType.GetEnumUnderlyingType()) : null,
                            Length: len,
                            Value: FieldValue(f, f.GetValue(inst)!),
                            CppValueText: CppFieldValueText(f, f.GetValue(inst)!),
                            Description: f.GetCustomAttribute<DescriptionAttribute>()!.Description,
                            IsEnum: f.PropertyType.IsEnum,
                            IsOpCode: f.PropertyType == typeof(OpCode)));
        }

        return fields.ToList();
    }

    private List<InstructionField> GetTensorInstructionFields(int index, Type t)
    {
        var props = new List<(int, PropertyInfo)>();
        var fields = new List<InstructionField>();
        foreach (var f in t.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        {
            int metadataToken = f.MetadataToken;
            props.Add((metadataToken, f));
        }

        props.Sort((a, b) => a.Item1 - b.Item1);

        fields.Add(new InstructionField(
                CppName: "opcode",
                CSharpName: "opcode",
                CSharpPropName: "OpCode",
                CppType: "opcode_t",
                CSharpType: "OpCode",
                UnderlyingCSharpType: CSharpFieldType(typeof(byte)),
                Length: FieldLength(typeof(byte)),
                Value: (uint)OpCode.TENSOR,
                CppValueText: null,
                Description: null,
                IsEnum: true,
                IsOpCode: true));

        fields.Add(new InstructionField(
                CppName: "tensor_funct",
                CSharpName: "tensorFunction",
                CSharpPropName: "TensorFunction",
                CppType: "tensor_function_t",
                CSharpType: "TensorFunction",
                UnderlyingCSharpType: CSharpFieldType(typeof(ushort)),
                Length: FieldLength(typeof(ushort)),
                Value: (uint)index,
                CppValueText: null,
                Description: null,
                IsEnum: true,
                IsOpCode: true));

        foreach (var (m, f) in props)
        {
            if (f.PropertyType.IsEnum)
            {
                GenerateEnum(f.PropertyType);
            }

            var len = FieldLength(f.PropertyType);
            fields.Add(new InstructionField(
                            CppName: SnakeName(f.Name),
                            CSharpName: CamelName(f.Name),
                            CSharpPropName: f.Name,
                            CppType: CppFieldType(f.PropertyType),
                            CSharpType: CSharpFieldType(f.PropertyType),
                            UnderlyingCSharpType: f.PropertyType.IsEnum ? CSharpFieldType(f.PropertyType.GetEnumUnderlyingType()) : null,
                            Length: len,
                            Value: null,
                            CppValueText: null,
                            Description: string.Empty,
                            IsEnum: f.PropertyType == typeof(Nncase.DataType) ? true : f.PropertyType.IsEnum,
                            IsOpCode: f.PropertyType == typeof(OpCode)));
        }

        return fields.ToList();
    }

    private List<InstructionInput> GetTensorInstructionInputs(Type t)
    {
        var props = new List<(int, FieldInfo)>();
        var fields = new List<InstructionInput>();
        foreach (var f in t.GetFields(BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly))
        {
            int metadataToken = f.MetadataToken;
            props.Add((metadataToken, f));
        }

        props.Sort((a, b) => a.Item1 - b.Item1);

        foreach (var (m, f) in props)
        {
            fields.Add(new InstructionInput(
                            CppType: "value_t",
                            CppName: SnakeName(f.Name)));
        }

        return fields.ToList();
    }

    private string? CppFieldValueText(PropertyInfo f, object v)
    {
        if (f.SetMethod != null)
        {
            return null;
        }

        if (f.PropertyType.IsEnum)
        {
            return CppFieldType(f.PropertyType) + "::" + v.ToString();
        }

        return v.ToString()!;
    }

    private uint FieldLength(Type t)
    {
        if (t.IsEnum)
        {
            return FieldLength(t.GetEnumUnderlyingType());
        }
        else if (t == typeof(bool))
        {
            return 8;
        }
        else if (t == typeof(byte))
        {
            return 8;
        }
        else if (t == typeof(ushort))
        {
            return 16;
        }
        else if (t == typeof(uint))
        {
            return 32;
        }
        else if (t == typeof(short))
        {
            return 16;
        }
        else if (t == typeof(int))
        {
            return 32;
        }
        else if (t == typeof(float))
        {
            return 32;
        }
        else if (t == typeof(Nncase.DataType))
        {
            return 8;
        }
        else
        {
            return 0;
        }
    }

    private string CppFieldType(Type t)
    {
        if (t == typeof(bool))
        {
            return "bool";
        }
        else if (t == typeof(byte))
        {
            return "uint8_t";
        }
        else if (t == typeof(ushort))
        {
            return "uint16_t";
        }
        else if (t == typeof(uint))
        {
            return "uint32_t";
        }
        else if (t == typeof(short))
        {
            return "int16_t";
        }
        else if (t == typeof(int))
        {
            return "int32_t";
        }
        else if (t == typeof(float))
        {
            return "float";
        }
        else if (t == typeof(Nncase.DataType))
        {
            return "typecode_t";
        }
        else if (t == typeof(byte[]))
        {
            return "std::span<const std::byte>";
        }
        else if (t == typeof(string))
        {
            return "std::string";
        }
        else if (t == typeof(string[]))
        {
            return "std::vector<std::string>";
        }
        else
        {
            return SnakeTypeName(t.Name);
        }
    }

    private string CSharpFieldType(Type t)
    {
        if (t == typeof(bool))
        {
            return "bool";
        }
        else if (t == typeof(byte))
        {
            return "byte";
        }
        else if (t == typeof(ushort))
        {
            return "ushort";
        }
        else if (t == typeof(uint))
        {
            return "uint";
        }
        else if (t == typeof(short))
        {
            return "short";
        }
        else if (t == typeof(int))
        {
            return "int";
        }
        else if (t == typeof(float))
        {
            return "float";
        }
        else if (t == typeof(Nncase.DataType))
        {
            return "DataType";
        }
        else if (t == typeof(string))
        {
            return "string";
        }
        else if (t == typeof(string[]))
        {
            return "string[]";
        }
        else
        {
            return PascalName(t.Name);
        }
    }

    private uint? FieldValue(PropertyInfo f, object v)
    {
        if (f.SetMethod != null)
        {
            return null;
        }

        var t = f.PropertyType;
        if (t.IsEnum)
        {
            return Convert.ToUInt32(v);
        }

        // Bits
        else
        {
            return uint.Parse(v.ToString()!);
        }
    }
}

internal class Program
{
    private static async Task Main(string[] args)
    {
        var ex = new IsaExtractor();
        ex.Extract();

        var inst_h = await ex.RenderAsync("Templates.opcode_h");
        File.WriteAllText(Path.Combine(args[0], "src/Native/include/nncase/runtime/stackvm", "opcode.h"), inst_h);

        var opreader_h = await ex.RenderAsync("Templates.op_reader_h");
        File.WriteAllText(Path.Combine(args[0], "src/Native/include/nncase/runtime/stackvm", "op_reader.h"), opreader_h);

        var opreader_cpp = await ex.RenderAsync("Templates.op_reader_cpp");
        File.WriteAllText(Path.Combine(args[0], "src/Native/src/runtime/stackvm", "op_reader.cpp"), opreader_cpp);

        var kernel_h = await ex.RenderAsync("Templates.kernel_h");
        File.WriteAllText(Path.Combine(args[0], "src/Native/include/nncase/kernels/stackvm", "tensor_ops.h"), kernel_h);

        var runtime_function_ops_h = await ex.RenderAsync("Templates.runtime_function_ops_h");
        File.WriteAllText(Path.Combine(args[0], "src/Native/src/runtime/stackvm", "runtime_function_ops.h"), runtime_function_ops_h);

        var runtime_function_tensor_ops_cpp = await ex.RenderAsync("Templates.runtime_function_tensor_ops_cpp");
        File.WriteAllText(Path.Combine(args[0], "src/Native/src/runtime/stackvm/ops", "tensor.cpp"), runtime_function_tensor_ops_cpp);

        var emitter_cs = await ex.RenderAsync("Templates.emitter_cs");
        File.WriteAllText(Path.Combine(args[0], "modules/Nncase.Modules.StackVM/CodeGen/StackVM", "StackVMEmitter.g.cs"), emitter_cs);

        var codegen_cs = await ex.RenderAsync("Templates.codegen_cs");
        File.WriteAllText(Path.Combine(args[0], "modules/Nncase.Modules.StackVM/CodeGen/StackVM", "CodeGenVisitor.g.cs"), codegen_cs);
    }
}

public record InstructionField(string CppName, string CSharpName, string CSharpPropName, string CppType, string CSharpType, string? UnderlyingCSharpType, uint Length, uint? Value, string? CppValueText, string? Description, bool IsEnum, bool IsOpCode);

public record InstructionInput(string CppType, string CppName);

public record InstructionInfo(int Index, string CppName, string CSharpName, string Category, OpCode OpCode, string Description, List<InstructionField> Fields, List<InstructionInput> Inputs);

public record EnumFieldInfo(string CppName, uint Value, string Description);

public record EnumInfo(string CppName, string UnderlyingCppType, string UnderlyingCSharpType, uint Length, List<EnumFieldInfo> Fields);
