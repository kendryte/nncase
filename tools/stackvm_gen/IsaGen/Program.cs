using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
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

            var opwriter_h = await ex.RenderAsync("Templates.op_writer_h");
            File.WriteAllText(Path.Combine(args[0], "include/nncase/codegen/stackvm", "op_writer.h"), opwriter_h);

            var opwriter_cpp = await ex.RenderAsync("Templates.op_writer_cpp");
            File.WriteAllText(Path.Combine(args[0], "src/codegen/stackvm", "op_writer.cpp"), opwriter_cpp);
        }
    }

    public class IsaExtractor
    {
        private readonly RazorLightEngine _engine;
        private readonly List<Instruction> _insts;
        private readonly List<Type> _enums;

        public IReadOnlyList<KeyValuePair<string, IReadOnlyList<InstructionInfo>>> Instructions { get; private set; }

        public IReadOnlyList<EnumInfo> Enums { get; private set; }

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
                                Length: (uint)(from f in fs select (int)f.Length).Sum(),
                                Description: t.GetCustomAttribute<DescriptionAttribute>().Description,
                                Fields: fs
                            ) by c).Select(x => new KeyValuePair<string, IReadOnlyList<InstructionInfo>>(x.Key, x.ToList())).ToList();

            Enums = (from e in _enums
                     let b = e.GetCustomAttribute<BrowsableAttribute>()
                     where b == null || b.Browsable
                     select new EnumInfo
                     (
                         Name: e.GetCustomAttribute<EnumNameAttribute>().Name,
                         Length: e.GetCustomAttribute<BitLengthAttribute>().BitLength,
                         Fields: GetEnumFields(e)
                     )).ToList();
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
                        Name: f.Name,
                        Value: (uint)(int)f.GetValue(null),
                        Description: f.GetCustomAttribute<DescriptionAttribute>()?.Description ?? string.Empty
                    )).ToList();
        }

        private List<InstructionField> GetInstructionFields(Instruction inst, Type t)
        {
            var props = new List<(int, PropertyInfo)>();
            var fields = new List<InstructionField>();
            uint start = 0;
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
                                Start: start,
                                Length: len,
                                End: start + len - 1,
                                Value: FieldValue(f, f.GetValue(inst)),
                                ValueText: FieldValueText(f, f.GetValue(inst)),
                                Description: f.GetCustomAttribute<DescriptionAttribute>().Description,
                                IsEnum: f.PropertyType.IsEnum
                            ));
                start += len;
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
                return t.GetCustomAttribute<BitLengthAttribute>().BitLength;
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
            // Bits
            else
                return uint.Parse(t.Name.Substring(3));
        }

        private string FieldType(Type t)
        {
            if (t.IsEnum)
                return t.GetCustomAttribute<EnumNameAttribute>().Name;
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
            // Bits
            else
                return t.Name.ToUpperInvariant();
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

    public record InstructionField(string Name, string Type, uint Start, uint Length, uint End, uint? Value, string ValueText, string Description, bool IsEnum);

    public record InstructionInfo(string Name, string Category, OpCode OpCode, uint Length, string Description, List<InstructionField> Fields);

    public record EnumFieldInfo(string Name, uint Value, string Description);

    public record EnumInfo(string Name, uint Length, List<EnumFieldInfo> Fields);
}
