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

namespace IsaGen
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var ex = new IsaExtractor();
            ex.Extract();

            var inst_h = await ex.RenderAsync("Templates.opcode_h");
            File.WriteAllText(Path.Combine(args[0], "include/nncase/runtime/stackvm", "opcode.h"), inst_h);

            var opreader_h = await ex.RenderAsync("Templates.op_reader_h");
            File.WriteAllText(Path.Combine(args[0], "include/nncase/runtime/stackvm", "op_reader.h"), opreader_h);

            var opreader_cpp = await ex.RenderAsync("Templates.op_reader_cpp");
            File.WriteAllText(Path.Combine(args[0], "src/runtime/stackvm", "op_reader.cpp"), opreader_cpp);

            //var opwriter_h = await ex.RenderAsync("Templates.op_writer_h");
            //File.WriteAllText(Path.Combine(args[0], "include/nncase/codegen/stackvm", "op_writer.h"), opwriter_h);

            //var opwriter_cpp = await ex.RenderAsync("Templates.op_writer_cpp");
            //File.WriteAllText(Path.Combine(args[0], "src/codegen/stackvm", "op_writer.cpp"), opwriter_cpp);
        }
    }

    public class IsaExtractor
    {
        private readonly RazorLightEngine _engine;
        private readonly List<Instruction> _insts;
        private readonly List<Type> _tensorInsts;
        private readonly List<Type> _enums;
        private readonly HashSet<Type> _generatedEnums = new HashSet<Type>();

        public IReadOnlyList<KeyValuePair<string, IReadOnlyList<InstructionInfo>>> Instructions { get; private set; }

        public IReadOnlyList<KeyValuePair<string, IReadOnlyList<InstructionInfo>>> TensorInstructions { get; private set; }

        public List<EnumInfo> Enums { get; private set; }

        public IsaExtractor()
        {
            _engine = new RazorLightEngineBuilder()
                .UseMemoryCachingProvider()
                .UseProject(new EmbeddedRazorProject(typeof(Program)) { Extension = ".razor" })
                .Build();

            _insts = (from t in typeof(Instruction).Assembly.ExportedTypes
                      where !t.IsAbstract && t.IsAssignableTo(typeof(Instruction))
                      orderby t.MetadataToken
                      select (Instruction)Activator.CreateInstance(t)).ToList();
            _enums = (from t in typeof(Instruction).Assembly.ExportedTypes
                      where t.IsEnum && t.IsDefined(typeof(EnumNameAttribute))
                      orderby t.MetadataToken
                      select t).ToList();
            _tensorInsts = (from t in typeof(Nncase.CoreModule).Assembly.ExportedTypes
                            where !t.IsAbstract && t.IsAssignableTo(typeof(Nncase.IR.Op))
                            select t).ToList();
        }

        public void Extract()
        {
            Instructions = (from inst in _insts
                            let t = inst.GetType()
                            let c = t.GetCustomAttribute<CategoryAttribute>().Category
                            let fs = GetInstructionFields(inst, t)
                            group new InstructionInfo
                            (
                                Name: t.GetCustomAttribute<DisplayNameAttribute>().DisplayName,
                                Category: c,
                                OpCode: inst.OpCode,
                                Description: t.GetCustomAttribute<DescriptionAttribute>().Description,
                                Fields: fs
                            ) by c).Select(x => new KeyValuePair<string, IReadOnlyList<InstructionInfo>>(x.Key, x.ToList())).ToList();


            Enums = (from e in _enums
                     let b = e.GetCustomAttribute<BrowsableAttribute>()
                     where b == null || b.Browsable
                     select new EnumInfo
                     (
                         Name: e.GetCustomAttribute<EnumNameAttribute>().Name,
                         UnderlyingType: FieldType(e.GetEnumUnderlyingType()),
                         Length: FieldLength(e),
                         Fields: GetEnumFields(e)
                     )).ToList();

            TensorInstructions = (from t in _tensorInsts
                                  where t.Namespace.StartsWith("Nncase.IR.")
                                  let c = t.Namespace.Replace("Nncase.IR.", string.Empty).Replace('.', '_').ToLowerInvariant()
                                  where c != "buffer"
                                  let fs = GetTensorInstructionFields(t)
                                  group new InstructionInfo
                                  (
                                      Name: SnakeName(t.Name),
                                      Category: c,
                                      OpCode: OpCode.TENSOR,
                                      Description: string.Empty,
                                      Fields: fs
                                  ) by c).Select(x => new KeyValuePair<string, IReadOnlyList<InstructionInfo>>(x.Key, x.ToList())).ToList();

            AddTensorFunctionEnum();

        }

        private void AddTensorFunctionEnum()
        {
            var fields = (from t in TensorInstructions.SelectMany(x => x.Value).OrderBy(x => x.Name).Select((x, i) => (x, i))
                          select new EnumFieldInfo
                          (
                              Name: t.x.Name,
                              Value: (uint)t.i,
                              Description: string.Empty
                          )).ToList();
            var e = new EnumInfo
                (
                    Name: "tensor_function_t",
                    UnderlyingType: FieldType(typeof(ushort)),
                    Length: FieldLength(typeof(ushort)),
                    Fields: fields
                );
            Enums.Insert(Enums.FindIndex(x => x.Name == "opcode_t") + 1, e);
        }

        public Task<string> RenderAsync(string templateName)
        {
            return _engine.CompileRenderAsync(templateName, this);
        }

        private List<EnumFieldInfo> GetEnumFields(Type e)
        {
            return (from f in e.GetFields(BindingFlags.Public | BindingFlags.Static)
                    select new EnumFieldInfo
                    (
                        Name: e == typeof(OpCode) ? f.Name : SnakeName(f.Name),
                        Value: (uint)Convert.ToInt32(f.GetValue(null)),
                        Description: f.GetCustomAttribute<DescriptionAttribute>()?.Description ?? string.Empty
                    )).ToList();
        }

        private string SnakeName(string name)
        {
            if (name == "OpCode")
                return "opcode";

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
                        sb.Append('_');
                }
                sb.Append(char.ToLowerInvariant(c));

                if (!lastIsLetter && c == 'D')
                    sb.Append('_');

                lastCapital = isCaptial;
                lastIsLetter = isLetter;
            }
            return sb.ToString().Trim('_');
        }

        private string SnakeTypeName(string name)
        {
            return SnakeName(name) + "_t";
        }

        private void GenerateEnum(Type type)
        {
            if (_generatedEnums.Contains(type)) return;
            _generatedEnums.Add(type);
            Enums.Add(new EnumInfo
                (
                    Name: SnakeTypeName(type.Name),
                    UnderlyingType: FieldType(type.GetEnumUnderlyingType()),
                    Length: FieldLength(type),
                    Fields: GetEnumFields(type)
                ));
        }

        private List<InstructionField> GetInstructionFields(Instruction inst, Type t)
        {
            var props = new List<(int, PropertyInfo)>();
            var fields = new List<InstructionField>();
            foreach (var f in t.GetProperties())
            {
                int metadataToken = f.MetadataToken;
                var baseType = t.BaseType;
                while (baseType != typeof(Instruction))
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
                fields.Add(new InstructionField
                            (
                                Name: f.GetCustomAttribute<DisplayNameAttribute>().DisplayName,
                                Type: FieldType(f.PropertyType),
                                Length: len,
                                Value: FieldValue(f, f.GetValue(inst)),
                                ValueText: FieldValueText(f, f.GetValue(inst)),
                                Description: f.GetCustomAttribute<DescriptionAttribute>().Description,
                                IsEnum: f.PropertyType.IsEnum,
                                IsOpCode: f.PropertyType == typeof(OpCode)
                            ));
            }

            return fields.ToList();
        }

        private List<InstructionField> GetTensorInstructionFields(Type t)
        {
            var props = new List<(int, PropertyInfo)>();
            var fields = new List<InstructionField>();
            foreach (var f in t.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                int metadataToken = f.MetadataToken;
                props.Add((metadataToken, f));
            }

            props.Sort((a, b) => a.Item1 - b.Item1);

            foreach (var (m, f) in props)
            {
                if (f.PropertyType.IsEnum)
                    GenerateEnum(f.PropertyType);

                var len = FieldLength(f.PropertyType);
                fields.Add(new InstructionField
                            (
                                Name: SnakeName(f.Name),
                                Type: FieldType(f.PropertyType),
                                Length: len,
                                Value: null,
                                ValueText: null,
                                Description: string.Empty,
                                IsEnum: f.PropertyType == typeof(Nncase.DataType) ? true : f.PropertyType.IsEnum,
                                IsOpCode: f.PropertyType == typeof(OpCode)
                            ));
            }

            return fields.ToList();
        }

        private string FieldValueText(PropertyInfo f, object v)
        {
            if (f.SetMethod != null)
                return null;
            if (f.PropertyType.IsEnum)
                return FieldType(f.PropertyType) + "::" + v.ToString();
            return v.ToString();
        }

        private uint FieldLength(Type t)
        {
            if (t.IsEnum)
                return FieldLength(t.GetEnumUnderlyingType());
            else if (t == typeof(bool))
                return 8;
            else if (t == typeof(byte))
                return 8;
            else if (t == typeof(ushort))
                return 16;
            else if (t == typeof(uint))
                return 32;
            else if (t == typeof(short))
                return 16;
            else if (t == typeof(int))
                return 32;
            else if (t == typeof(float))
                return 32;
            else if (t == typeof(Nncase.DataType))
                return 8;
            else
                return 0;
        }

        private string FieldType(Type t)
        {
            if (t == typeof(bool))
                return "bool";
            else if (t == typeof(byte))
                return "uint8_t";
            else if (t == typeof(ushort))
                return "uint16_t";
            else if (t == typeof(uint))
                return "uint32_t";
            else if (t == typeof(short))
                return "int16_t";
            else if (t == typeof(int))
                return "int32_t";
            else if (t == typeof(float))
                return "float";
            else if (t == typeof(Nncase.DataType))
                return "typecode_t";
            else
                return SnakeTypeName(t.Name);
        }

        private uint? FieldValue(PropertyInfo f, object v)
        {
            if (f.SetMethod != null)
                return null;

            var t = f.PropertyType;
            if (t.IsEnum)
                return Convert.ToUInt32(v);
            // Bits
            else
                return uint.Parse(v.ToString());
        }

        public static string ToBinaryString(uint value, uint bits)
        {
            return Convert.ToString(value, 2).PadLeft((int)bits, '0');
        }
    }

    public record InstructionField(string Name, string Type, uint Length, uint? Value, string ValueText, string Description, bool IsEnum, bool IsOpCode);

    public record InstructionInfo(string Name, string Category, OpCode OpCode, string Description, List<InstructionField> Fields);

    public record EnumFieldInfo(string Name, uint Value, string Description);

    public record EnumInfo(string Name, string UnderlyingType, uint Length, List<EnumFieldInfo> Fields);
}
