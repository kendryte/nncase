using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IO;

namespace Nncase.CodeGen;


public interface SectionDecompiler
{
    void Decompile(ReadOnlySpan<byte> input, ReadOnlySpan<Symbol> symbols, Stream ostream);
}

public struct FunctionCallId
{
    public int ModuleId;
    public int FunctionId;
}

/// <summary>
/// the kmodel's module define
/// </summary>
public abstract class BaseRTKModule : IRTModule
{

    protected class Section
    {
        public readonly SectionWriter Writer;
        public readonly MemoryStream Output;
        public byte[] Body;
        public Section()
        {
            Output = new();
            Writer = new(Output);
            Body = new byte[0];
        }
    };

    protected struct RdataMergeInfo
    {
        public ulong Start;
        public ulong Size;
    }


    public BaseRTKModule(uint alignment, string module_name,
             Schedule.SchedModuleResult ModuleResult,
            Schedule.SchedModelResult modelResult)
    {
        _alignment = alignment;
        _moduleName = module_name;
        _modelResult = modelResult;
        _moduleResult = ModuleResult;
    }

    public uint Alignment => _alignment;

    // todo public void config_dump(const std::filesystem::path &dump_dir, bool dump_asm); 
    public void Build(BinaryWriter writer) { }

    public Schedule.BufferAllocation Allocation(IR.Expr conn)
    {
        return _moduleResult.Allocations[conn];
    }

    public ulong MaxUsage(Schedule.MemoryLocation location)
    {
        if (_moduleResult.MaxUsages.TryGetValue(location, out var value))
        {
            return value;
        }
        return 0;
    }

    public SectionWriter Writer(string section_name)
    {
        if (!_sectionWriters.TryGetValue(section_name, out var section))
        {
            section = new();
            _sectionWriters.Add(section_name, section);
        }
        return section.Writer;
    }

    public ModuleType ModuleType { get; set; }
    public abstract uint ModuleVersion { get; }
    public abstract string Source { get; set; }
    public abstract string SourceExt { get; set; }
    public abstract IReadOnlyList<IRTFunction> Functions { get; }

    public abstract SectionDecompiler CreateDecompiler(string section_name);
    protected Section? FindSection(string section_name)
    {
        if (!_sectionWriters.TryGetValue(section_name, out var section))
        {
            return null;
        }
        return section;
    }
    protected void MergeToRdataSection(string from) { }
    protected FunctionCallId FunctionId(IR.Expr expr)
    {
        for (int i = 0; i < _modelResult.Modules.Count; i++)
        {
            var mod_sched = _modelResult.Modules[i];
            if (mod_sched.FunctionsMap.TryGetValue(expr, out var func_it))
            {
                if (mod_sched.Functions.IndexOf(func_it) is var idx)
                    return new FunctionCallId() { ModuleId = i, FunctionId = idx };
            }
        }
        throw new InvalidProgramException("Can't find expr in modules");
    }
    protected void SetCurrentEntryPoint(long pos)
    {
        _entryPoints[_currentFunction] = (ulong)pos;
    }
    protected void SetCurrentFunctionTextEnd(long pos)
    {
        _functionTextEnd[_currentFunction] = (ulong)pos;
    }

    // for each module custom implment
    protected abstract void BeginEmitModule();
    protected abstract void BeginEmitFunction(Schedule.SchedFunctionResult function);
    protected abstract void EndEmitFunction(Schedule.SchedFunctionResult function);
    protected abstract void Emit(IR.Expr node);
    protected abstract void EndEmitModule();


    protected static HashSet<RuntimeTypeHandle> s_nonRuntimeOps = new() { typeof(IR.Var).TypeHandle };
    private List<IR.Expr> GenerateCurrentRuntimeOps()
    {
        List<IR.Expr> runtime_ops = new();
        foreach (var item in _currentFunction.ComputeSequence)
        {
            if (!s_nonRuntimeOps.Contains(item.GetType().TypeHandle))
                runtime_ops.Add(item);
        }
        return runtime_ops;
    }

    private void Compile()
    {
        WriteConstants();
        BeginEmitModule();
        foreach (var func_sched in _moduleResult.Functions)
        {
            _currentFunction = func_sched;
            var runtime_ops = GenerateCurrentRuntimeOps();
            BeginEmitFunction(_currentFunction);
            foreach (var item in runtime_ops)
            {
                Emit(item);
            }
            EndEmitFunction(_currentFunction);

            if (!_entryPoints.ContainsKey(_currentFunction))
                throw new InvalidProgramException($"Entry point is not set");
        }
        EndEmitModule();
        // if (dump_asm_)
        // {
        //     for (auto &section : section_writer_)
        //         section.second.body = read_stream(section.second.stream);

        //     for (auto &section : section_writer_)
        //         decompile("compile", section.first, section.second.body, section.second.writer.symbols());
        // }
    }

    private void Decompile(string stage, string section_name, ReadOnlySpan<byte> input, ReadOnlySpan<Symbol> symbols)
    {
        if (CreateDecompiler(section_name) is var decompiler && decompiler is not null)
        {
            var ostream = new MemoryStream();
            decompiler.Decompile(input, symbols, ostream);
        }
        else
        {
            Console.WriteLine($"WARN: Cannot Find a Decompiler For Section {section_name}!");
        }
    }

    private void WriteConstants()
    {
        if (_moduleResult.MaxUsages.TryGetValue(Schedule.MemoryLocation.Rdata, out var useage))
        {
            var constants = new byte[useage];
            foreach (var func_sched in _moduleResult.Functions)
            {
                foreach (var item in func_sched.ComputeSequence)
                {
                    if (item is IR.Const con)
                    {
                        var alloc = Allocation(con);
                        con.Data.Array.CopyTo(constants.AsSpan((int)alloc.Start));
                    }
                }
            }
            Writer(".rdata").Write(constants);
        }
    }
    private void GenerateMergeInfo()
    {
        if (_rdataSectionMerges.Any())
        {
            var rdata_writer = Writer(".rdata");
            foreach (var merge_p in _rdataSectionMerges)
            {
                if (_sectionWriters.TryGetValue(merge_p.Key, out var section))
                {
                    rdata_writer.AlignPosition(_alignment);
                    var start = rdata_writer.BaseStream.Position;
                    rdata_writer.Write(section.Output.ToArray());
                    var size = rdata_writer.BaseStream.Position - start;
                    _rdataSectionMerges[merge_p.Key] = new RdataMergeInfo
                    {
                        Start = (ulong)start,
                        Size = (ulong)size
                    };
                }
            }
        }

        foreach (var section in _sectionWriters)
        {
            if (!_rdataSectionMerges.ContainsKey(section.Key))
                section.Value.Body = section.Value.Output.ToArray();
        }
        // if (dump_asm_)
        // {
        //     std::ofstream file(dump_dir_ / "section-merge.txt");
        //     for (auto &off : rdata_section_merges_)
        //         file << off.first << " = " << off.second.start << "@.rdata" << std::endl;
        // }
    }

    private void GenerateSymbolOffsets()
    {
        foreach (var section in _sectionWriters)
        {
            if (_rdataSectionMerges.TryGetValue(section.Key, out var info))
            {
                var section_start = info.Start;
                foreach (var symbol in section.Value.Writer.Symbols)
                {
                    _symbolOffsets.Add(symbol.Name, (section_start + (ulong)symbol.Streampos, ".rdata"));
                }
            }
            else
            {
                foreach (var symbol in section.Value.Writer.Symbols)
                {
                    _symbolOffsets.Add(symbol.Name, ((ulong)symbol.Streampos, section.Key));
                }
            }
        }

        // if (dump_asm_)
        // {
        //     std::ofstream file(dump_dir_ / "symbol-addr.txt");
        //     for (auto &off : symbol_offsets_)
        //         file << off.first << " = " << off.second.first << "@" << off.second.second << std::endl;
        // }
    }
    private void WriteSymbolRefs()
    {
        var rdata_writer = _sectionWriters[".rdata"];
        foreach (var section in _sectionWriters)
        {
            foreach (var refs in section.Value.Writer.SymbolRefs)
            {
                Span<byte> srcSpan;
                if (_rdataSectionMerges.TryGetValue(section.Key, out var info))
                {
                    srcSpan = rdata_writer.Body.AsSpan((int)info.Start, (int)info.Size);
                }
                else
                {
                    srcSpan = section.Value.Body;
                }

                var subSpan = srcSpan.Slice((int)refs.Streampos);
                var bw = new BitWriter(subSpan, refs.Bitoffset);
                bw.Write(_symbolOffsets[refs.Name].Offset, (int)refs.Length);
            }
        }
    }
    private void Link()
    {
        GenerateMergeInfo();
        GenerateSymbolOffsets();
        WriteSymbolRefs();
        // if (dump_asm_)
        // {
        //     for (auto &section : section_writer_)
        //     {
        //         if (rdata_section_merges_.contains(section.first))
        //             decompile("link", section.first, section.second.body, section.second.writer.symbols());
        //     }
        // }
    }
    private void WriteBinary(BinaryWriter writer)
    {
        // Step 1. skip the module header
        var header_pos = writer.BaseStream.Position;
        writer.Seek(Marshal.SizeOf(typeof(ModuleHeader)), SeekOrigin.Current);

        // mempools
        foreach (var mem in _moduleResult.MaxUsages)
        {
            var desc = new MemPoolDesc { Location = mem.Key, Size = (uint)mem.Value };
            writer.Write(CodeGenUtil.StructToBytes(desc));
        }

        // functions
        foreach (var func_sched in _moduleResult.Functions)
        {
            WriteFunctionBinary(writer, func_sched);
        }

        // sections
        foreach (var section in _sectionWriters)
        {
            var sec_header = new SectionHeader();
            section.Key.ToArray().CopyTo(sec_header.Name.AsSpan());

            if (_rdataSectionMerges.TryGetValue(section.Key, out var merge_it) is var finded && finded == false)
            {
                sec_header.Flags = 0;
                sec_header.BodyStart = 0;
                sec_header.BodySize = (uint)section.Value.Body.Length;
            }
            else
            {
                sec_header.Flags = ModelInfo.SECTION_MERGED_INTO_RDATA;
                sec_header.BodyStart = (uint)merge_it.Start;
                sec_header.BodySize = (uint)merge_it.Size;
            }

            // Skip section sec_header
            var sec_header_pos = writer.BaseStream.Position;
            writer.Seek(Marshal.SizeOf(sec_header), SeekOrigin.Current);

            if (finded is false)
            {
                sec_header.BodyStart = (uint)writer.AlignPosition(_alignment);
                writer.Write(section.Value.Body);// write content
            }

            // write section sec_header
            var sec_end_pos = writer.BaseStream.Position;
            writer.Seek((int)sec_header_pos, SeekOrigin.Begin);
            writer.Write(CodeGenUtil.StructToBytes(sec_header));
            writer.Seek((int)sec_end_pos, SeekOrigin.Begin);
        }

        writer.AlignPosition(8);
        var end_pos = writer.Position();

        // defin module_header
        var header = new ModuleHeader { };
        header.Type = ModuleType;
        header.Version = (uint)ModuleVersion;
        header.HeaderSize = (uint)Marshal.SizeOf<ModuleHeader>();
        header.Size = (uint)(end_pos - header_pos);
        header.Mempools = (uint)_moduleResult.MaxUsages.Count;
        header.SharedMempools = (uint)_moduleResult.SharedMaxUsages.Count;
        header.Functions = (uint)_moduleResult.Functions.Count;
        header.Sections = (uint)_sectionWriters.Count;
        header.Reserved0 = 0;
        writer.Position(header_pos);
        writer.Write(CodeGenUtil.StructToBytes(header));
        writer.Position(end_pos);
    }

    private void WriteFunctionBinary(BinaryWriter writer, Schedule.SchedFunctionResult function_sched)
    {
        void writeShape(IR.Shape shape)
        {
            writer.Write((uint)shape.Count);
            foreach (var dim in shape)
            {
                writer.Write((uint)dim.FixedValue);
            }
        }

        List<Schedule.MemoryRange> inputs = new();
        List<IR.Shape> input_shapes = new();
        List<Schedule.MemoryRange> outputs = new();
        List<IR.Shape> output_shapes = new();

        foreach (var node in function_sched.ComputeSequence)
        {
            if (function_sched.Function.Parameters.Contains(node,
                            new IR.RecordRefComparer<IR.Expr>()))
            {
                var alloc = Allocation(node);
                inputs.Add(alloc.RuntimeType);
                input_shapes.Add(alloc.Shape);
            }
            else if (object.ReferenceEquals(function_sched.Function.Body, node))
            {
                var alloc = Allocation(node);
                outputs.Add(alloc.RuntimeType);
                output_shapes.Add(alloc.Shape);
            }
        }

        // Skip function header
        var header_pos = writer.Position();
        writer.Skip((ulong)Marshal.SizeOf<FunctionHeader>());

        // inputs
        writer.Write(CodeGenUtil.StructToBytes(inputs));
        foreach (var shape in input_shapes)
            writeShape(shape);

        // outputs
        writer.Write(CodeGenUtil.StructToBytes(outputs));
        foreach (var shape in output_shapes)
            writeShape(shape);

        writer.AlignPosition(8);
        var end_pos = writer.Position();

        // header
        var header = new FunctionHeader
        {
            HeaderSize = (uint)Marshal.SizeOf<FunctionHeader>(),
            Size = (uint)(end_pos - header_pos),
            InputPoolSize = (uint)function_sched.InputPoolSize,
            OutputPoolSize = (uint)function_sched.OutputPoolSize,
            Inputs = (uint)inputs.Count,
            Outputs = (uint)outputs.Count,
            Entrypoint = (uint)_entryPoints[function_sched],
            TextSize = (uint)(_functionTextEnd[function_sched] - _entryPoints[function_sched]),
        };
        writer.Position(header_pos);
        writer.Write(CodeGenUtil.StructToBytes(header));
        writer.Position(end_pos);
    }

    public abstract void Dump(string name, string dumpDirPath);
    public abstract ISerializeResult Serialize();

    private uint _alignment;
    private string _moduleName;
    private readonly Schedule.SchedModelResult _modelResult;
    private readonly Schedule.SchedModuleResult _moduleResult;
    private Schedule.SchedFunctionResult _currentFunction;
    private readonly SortedDictionary<string, Section> _sectionWriters = new(StringComparer.CurrentCulture);
    private readonly SortedDictionary<string, RdataMergeInfo> _rdataSectionMerges = new(StringComparer.CurrentCulture);
    private readonly Dictionary<string, (ulong Offset, string Name)> _symbolOffsets;
    private readonly Dictionary<Schedule.SchedFunctionResult, ulong> _entryPoints;
    private readonly Dictionary<Schedule.SchedFunctionResult, ulong> _functionTextEnd;
}