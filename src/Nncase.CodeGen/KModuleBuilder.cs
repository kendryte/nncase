using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IO;

namespace Nncase.CodeGen;

/// <summary>
/// the section decompiler
/// </summary>
public interface ISectionDecompiler
{
    /// <summary>
    /// need impl by sub class
    /// </summary>
    /// <param name="input"></param>
    /// <param name="symbols"></param>
    /// <param name="ostream"></param>
    void Decompile(ReadOnlySpan<byte> input, ReadOnlySpan<Symbol> symbols, Stream ostream);
}

/// <summary>
/// save the funtion index
/// </summary>
public struct FunctionCallId
{
    /// <summary>
    /// module index
    /// </summary>
    public int ModuleId;
    /// <summary>
    /// function index
    /// </summary>
    public int FunctionId;
}

/// <summary>
/// the kmodel's module base class define
/// </summary>
public abstract class BaseRTKModule : IRTModule
{

    /// <summary>
    /// each kmodule have one or more scetion
    /// </summary>
    protected class Section
    {
        /// <summary>
        /// writer
        /// </summary>
        public readonly SectionWriter Writer;
        /// <summary>
        /// the contents
        /// </summary>
        public readonly MemoryStream Output;
        /// <summary>
        /// the byte of contents
        /// </summary>
        public byte[] Body;
        /// <summary>
        /// builder
        /// </summary>
        public Section()
        {
            Output = new();
            Writer = new(Output);
            Body = new byte[0];
        }
    };

    /// <summary>
    /// merge info
    /// </summary>
    protected struct RdataMergeInfo
    {
        /// <summary>
        /// start index
        /// </summary>
        public ulong Start;
        /// <summary>
        /// length
        /// </summary>
        public ulong Size;
    }

    /// <summary>
    /// builder
    /// </summary>
    /// <param name="ModuleResult"></param>
    /// <param name="modelResult"></param>
    public BaseRTKModule(Schedule.SchedModuleResult ModuleResult,
            Schedule.SchedModelResult modelResult)
    {
        _modelResult = modelResult;
        _moduleResult = ModuleResult;
        _currentFunction = modelResult.Entry!;
        _symbolOffsets = new();
        _entryPoints = new();
        _functionTextEnd = new();
    }

    /// <summary>
    /// get the Alignment
    /// </summary>
    public abstract uint Alignment { get; }

    // todo public void config_dump(const std::filesystem::path &dump_dir, bool dump_asm); 

    /// <summary>
    /// allocation buffer for node, just get it from sched result.
    /// </summary>
    /// <param name="conn"></param>
    /// <returns></returns>
    public Schedule.BufferAllocation Allocation(IR.Expr conn)
    {
        return _moduleResult.Allocations[conn];
    }

    /// <summary>
    /// get max mem usage
    /// </summary>
    /// <param name="location"></param>
    /// <returns></returns>
    public ulong MaxUsage(Schedule.MemoryLocation location)
    {
        if (_moduleResult.MaxUsages.TryGetValue(location, out var value))
        {
            return value;
        }
        return 0;
    }

    /// <summary>
    /// give the section name get the writer
    /// </summary>
    /// <param name="section_name"></param>
    /// <returns></returns>
    public SectionWriter Writer(string section_name)
    {
        if (!_sectionWriters.TryGetValue(section_name, out var section))
        {
            section = new();
            _sectionWriters.Add(section_name, section);
        }
        return section.Writer;
    }

    /// <inheritdoc/>
    public abstract ModuleType ModuleType { get; set; }
    /// <summary>
    /// the module verison
    /// </summary>
    public abstract uint ModuleVersion { get; }
    /// <inheritdoc/>
    public string Source { get; set; } = string.Empty;
    /// <inheritdoc/>
    public string SourceExt { get => "kmodule"; set { } }
    /// <inheritdoc/>
    public abstract IReadOnlyList<IRTFunction> Functions { get; }
    /// <summary>
    /// get the Decompiler
    /// </summary>
    /// <param name="section_name"></param>
    /// <returns></returns>
    public abstract ISectionDecompiler GetDecompiler(string section_name);
    /// <summary>
    /// get section by name
    /// </summary>
    /// <param name="section_name"></param>
    /// <returns></returns>
    protected Section? FindSection(string section_name)
    {
        if (!_sectionWriters.TryGetValue(section_name, out var section))
        {
            return null;
        }
        return section;
    }

    /// <summary>
    /// merget rdata section
    /// </summary>
    /// <param name="from"></param>
    protected void MergeToRdataSection(string from)
    {
        _rdataSectionMerges.Add(from, new());
    }

    /// <summary>
    /// get current function id.
    /// </summary>
    /// <param name="expr"></param>
    /// <returns></returns>
    /// <exception cref="InvalidProgramException"></exception>
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

    /// <summary>
    /// seth the entry function start pos 
    /// </summary>
    /// <param name="pos"></param>
    protected void SetCurrentEntryPoint(long pos)
    {
        _entryPoints[_currentFunction] = (ulong)pos;
    }

    /// <summary>
    /// set the current function end pos
    /// </summary>
    /// <param name="pos"></param>
    protected void SetCurrentFunctionTextEnd(long pos)
    {
        _functionTextEnd[_currentFunction] = (ulong)pos;
    }

    /// <summary>
    /// the callback BeginEmitModule
    /// </summary>
    protected abstract void BeginEmitModule();
    /// <summary>
    /// the callback BeginEmit func
    /// </summary>
    /// <param name="function"></param>
    protected abstract void BeginEmitFunction(Schedule.SchedFunctionResult function);
    /// <summary>
    /// the call back end emit func
    /// </summary>
    /// <param name="function"></param>
    protected abstract void EndEmitFunction(Schedule.SchedFunctionResult function);
    /// <summary>
    /// the emit 
    /// </summary>
    /// <param name="node"></param>
    protected abstract void Emit(IR.Function node);
    /// <summary>
    /// the call back end emit module 
    /// </summary>
    protected abstract void EndEmitModule();

    /// <summary>
    /// get the code section writer
    /// </summary>
    protected abstract SectionWriter TextWriter { get; }


    /// <summary>
    /// the dict contains the expr type which can not emit to binary.
    /// </summary>
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
            BeginEmitFunction(_currentFunction);
            Emit(_currentFunction.Function);
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
        if (GetDecompiler(section_name) is var decompiler && decompiler is not null)
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
                    rdata_writer.AlignPosition(Alignment);
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
        if (_sectionWriters.TryGetValue(".rdata", out var rdata_writer))
        {
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
            writer.Write(CodeGenUtil.StructToBytes<MemPoolDesc>(desc));
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
            writer.Seek(Marshal.SizeOf<SectionHeader>(), SeekOrigin.Current);

            if (finded is false)
            {
                sec_header.BodyStart = (uint)writer.AlignPosition(Alignment);
                writer.Write(section.Value.Body);// write content
            }

            // write section sec_header
            var sec_end_pos = writer.BaseStream.Position;
            writer.Seek((int)sec_header_pos, SeekOrigin.Begin);
            writer.Write(CodeGenUtil.StructToBytes<SectionHeader>(sec_header));
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
        writer.Write(CodeGenUtil.StructToBytes<ModuleHeader>(header));
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
        foreach (var input in inputs) { writer.Write(CodeGenUtil.StructToBytes<Schedule.MemoryRange>(input)); }
        foreach (var shape in input_shapes)
            writeShape(shape);

        // outputs
        foreach (var output in outputs) { writer.Write(CodeGenUtil.StructToBytes<Schedule.MemoryRange>(output)); }
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
        writer.Write(CodeGenUtil.StructToBytes<FunctionHeader>(header));
        writer.Position(end_pos);
    }

    /// <inheritdoc/>
    public void Dump(string name, string dumpDirPath)
    {
        if (!_isSerialized) Serialize();
        File.Copy(_sourcePath, Path.Join(dumpDirPath, name + '.' + SourceExt));
    }
    /// <inheritdoc/>
    public ISerializeResult Serialize()
    {
        if (!_isSerialized)
        {
            _sourcePath = CodeGenUtil.GetTempFileName(SourceExt);
            using var f = new FileStream(_sourcePath, FileMode.OpenOrCreate, FileAccess.ReadWrite);
            using var bw = new BinaryWriter(f, Encoding.Default, true);
            Compile();
            Link();
            WriteBinary(bw);
            _isSerialized = true;
        }
        return new KModuleSerializeResult(Alignment);
    }

    private bool _isSerialized = false;
    private string _sourcePath = string.Empty;
    private readonly Schedule.SchedModelResult _modelResult;
    private readonly Schedule.SchedModuleResult _moduleResult;
    private Schedule.SchedFunctionResult _currentFunction;
    private readonly SortedDictionary<string, Section> _sectionWriters = new(StringComparer.CurrentCulture);
    private readonly SortedDictionary<string, RdataMergeInfo> _rdataSectionMerges = new(StringComparer.CurrentCulture);
    private readonly Dictionary<string, (ulong Offset, string Name)> _symbolOffsets;
    private readonly Dictionary<Schedule.SchedFunctionResult, ulong> _entryPoints;
    private readonly Dictionary<Schedule.SchedFunctionResult, ulong> _functionTextEnd;
}