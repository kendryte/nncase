// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IO;

namespace Nncase.CodeGen;

/// <summary>
/// the section decompiler.
/// </summary>
public interface ISectionDecompiler
{
    /// <summary>
    /// need impl by sub class.
    /// </summary>
    /// <param name="input"></param>
    /// <param name="symbols"></param>
    /// <param name="ostream"></param>
    void Decompile(ReadOnlySpan<byte> input, ReadOnlySpan<Symbol> symbols, Stream ostream);
}

/// <summary>
/// save the funtion index.
/// </summary>
public struct FunctionCallId
{
    /// <summary>
    /// module index.
    /// </summary>
    public int ModuleId;

    /// <summary>
    /// function index.
    /// </summary>
    public int FunctionId;
}

/// <summary>
/// the kmodel's module base class define.
/// </summary>
public abstract class BaseRTKModule : IRTModule
{
    /// <summary>
    /// each kmodule have one or more scetion.
    /// </summary>
    protected class Section
    {
        /// <summary>
        /// writer.
        /// </summary>
        public readonly SectionWriter Writer;

        /// <summary>
        /// the contents.
        /// </summary>
        public readonly MemoryStream Output;

        /// <summary>
        /// the byte of contents.
        /// </summary>
        public byte[] Body;

        /// <summary>
        /// builder.
        /// </summary>
        public Section()
        {
            Output = new();
            Writer = new(Output);
            Body = new byte[0];
        }
    }



    /// <summary>
    /// merge info.
    /// </summary>
    protected struct RdataMergeInfo
    {
        /// <summary>
        /// start index.
        /// </summary>
        public ulong Start;

        /// <summary>
        /// length.
        /// </summary>
        public ulong Size;
    }

    /// <inheritdoc/>
    public bool IsSerialized { get; private set; }

    /// <summary>
    /// builder.
    /// </summary>
    public BaseRTKModule(IR.IRModel model, IR.IRModule module)
    {
        _model = model;
        _module = module;
        _currentFunction = model.Entry;
        _symbolOffsets = new();
        _entryPoints = new();
        _functionTextEnd = new();
    }

    /// <summary>
    /// get the Alignment.
    /// </summary>
    public abstract uint Alignment { get; }

    // todo public void config_dump(const std::filesystem::path &dump_dir, bool dump_asm);

    /// <summary>
    /// get max mem usage.
    /// </summary>
    /// <param name="location"></param>
    /// <returns></returns>
    public ulong MaxUsage(Schedule.MemoryLocation location)
    {
        if (_module.SchedResult!.MaxUsages.TryGetValue(location, out var value))
        {
            return value;
        }

        return 0;
    }

    /// <summary>
    /// give the section name get the writer.
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
    public abstract ModuleType ModuleType { get; }
    /// <summary>
    /// the module verison.
    /// </summary>
    public abstract uint ModuleVersion { get; }

    /// <inheritdoc/>
    public byte[] Source
    {
        get
        {
            if (IsSerialized)
                return File.ReadAllBytes(_sourcePath);
            throw new InvalidOperationException("Must Serialized Runtime Module Can Get The Source!");
        }
    }
    /// <inheritdoc/>
    public string SourceExt { get => "kmodule"; set { } }

    /// <inheritdoc/>
    public abstract IReadOnlyList<IRTFunction> Functions { get; }

    /// <summary>
    /// get the Decompiler.
    /// </summary>
    /// <param name="section_name"></param>
    /// <returns></returns>
    public abstract ISectionDecompiler GetDecompiler(string section_name);

    /// <summary>
    /// get section by name.
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
    /// merget rdata section.
    /// </summary>
    /// <param name="from"></param>
    protected void MergeToRdataSection(string from)
    {
        _rdataSectionMerges.Add(from, new());
    }

    /// <summary>
    /// get current function id.
    /// </summary>
    /// <param name="current_func"></param>
    /// <returns></returns>
    /// <exception cref="InvalidProgramException"></exception>
    protected FunctionCallId FunctionId(TIR.PrimFunction current_func)
    {
        for (int i = 0; i < _model.Modules.Count; i++)
        {
            var module = _model.Modules[i];
            for (int j = 0; j < module.Callables.Count; j++)
            {
                if (ReferenceEqualityComparer.Instance.Equals(current_func, module.Callables[j]))
                {
                    return new FunctionCallId() { ModuleId = i, FunctionId = j };
                }
            }
        }
        throw new InvalidProgramException("Can't find this func in the modules!");
    }

    /// <summary>
    /// seth the entry function start pos.
    /// </summary>
    /// <param name="pos"></param>
    protected void SetCurrentEntryPoint(long pos)
    {
        _entryPoints[_currentFunction] = (ulong)pos;
    }

    /// <summary>
    /// set the current function end pos.
    /// </summary>
    /// <param name="pos"></param>
    protected void SetCurrentFunctionTextEnd(long pos)
    {
        _functionTextEnd[_currentFunction] = (ulong)pos;
    }

    /// <summary>
    /// the callback BeginEmitModule.
    /// </summary>
    protected abstract void BeginEmitModule();

    /// <summary>
    /// the callback BeginEmit func.
    /// </summary>
    /// <param name="function"></param>
    protected abstract void BeginEmitFunction(IR.Callable function);

    /// <summary>
    /// the call back end emit func.
    /// </summary>
    /// <param name="function"></param>
    protected abstract void EndEmitFunction(IR.Callable function);

    /// <summary>
    /// the emit.
    /// </summary>
    /// <param name="node"></param>
    protected abstract void Emit(IR.Callable node);

    /// <summary>
    /// the call back end emit module.
    /// </summary>
    protected abstract void EndEmitModule();

    /// <summary>
    /// get the code section writer.
    /// </summary>
    protected abstract SectionWriter TextWriter { get; }

    /// <summary>
    /// the dict contains the expr type which can not emit to binary.
    /// </summary>
    protected static HashSet<RuntimeTypeHandle> s_nonRuntimeOps = new() { typeof(IR.Var).TypeHandle };

    private List<IR.Expr> GenerateCurrentRuntimeOps()
    {
        List<IR.Expr> runtime_ops = new();
        // todo refactor gen stackvm runtime ops.
        // foreach (var item in _currentFunction.ComputeSequence)
        // {
        //     if (!s_nonRuntimeOps.Contains(item.GetType().TypeHandle))
        //     {
        //         runtime_ops.Add(item);
        //     }
        // }

        return runtime_ops;
    }

    private void Compile()
    {
        WriteConstants();
        BeginEmitModule();
        foreach (var func in _module.Callables)
        {
            _currentFunction = func;
            BeginEmitFunction(_currentFunction);
            Emit(_currentFunction);
            EndEmitFunction(_currentFunction);

            if (!_entryPoints.ContainsKey(_currentFunction))
            {
                throw new InvalidProgramException($"Entry point is not set");
            }
        }
        EndEmitModule();

        // todo impl dump_asm
        // if (dump_asm_)
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
        //todo we need refactor the bufferlize.
        if (_module.SchedResult!.MaxUsages.TryGetValue(Schedule.MemoryLocation.Rdata, out var useage))
        {
            var constants = new byte[useage];

            foreach (var kv in (from func in _module.Callables
                                let sched = func.SchedResult
                                where sched is not null
                                from kv in sched.Allocations
                                where kv.Key.MemLocation == Schedule.MemoryLocation.Rdata
                                select kv))
            {
                var (buffer, allocate) = (kv.Key, kv.Value);
                ((IR.TensorConst)buffer.Const!).Value.BytesBuffer.CopyTo(constants.AsSpan((int)allocate.Start));
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
                        Size = (ulong)size,
                    };
                }
            }
        }

        foreach (var section in _sectionWriters)
        {
            if (!_rdataSectionMerges.ContainsKey(section.Key))
            {
                section.Value.Body = section.Value.Output.ToArray();
            }
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

        // todo refactor the dump.
        // if (dump_asm_)
        // {
        //     for (auto &section : section_writer_)
        //     {
        //         if (rdata_section_merges_.contains(section.first))
        //             decompile("link", section.first, section.second.body, section.second.writer.symbols());
        //     }
        // }
    }

    private unsafe void WriteBinary(BinaryWriter writer)
    {
        // Step 1. skip the module header
        var header_pos = writer.BaseStream.Position;
        writer.Seek(Marshal.SizeOf(typeof(ModuleHeader)), SeekOrigin.Current);

        // mempools
        foreach (var mem in _module.SchedResult!.MaxUsages)
        {
            var desc = new MemPoolDesc { Location = mem.Key, Size = (uint)mem.Value };
            writer.Write(CodeGenUtil.StructToBytes<MemPoolDesc>(desc));
        }

        // functions
        foreach (var func_sched in _module.Callables)
        {
            WriteCallableBinary(writer, func_sched);
        }

        // sections
        foreach (var section in _sectionWriters)
        {
            var sec_header = new SectionHeader();

            // for (int i = 0; i < section.Key.Length; i++) {  = section.Key[i]; }
            sec_header.Name = section.Key;

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
        header.Mempools = (uint)_module.SchedResult!.MaxUsages.Count;
        header.SharedMempools = (uint)_module.SchedResult!.SharedMaxUsages.Count;
        header.Functions = (uint)_module.Callables.Count;
        header.Sections = (uint)_sectionWriters.Count;
        header.Reserved0 = 0;
        writer.Position(header_pos);
        writer.Write(CodeGenUtil.StructToBytes<ModuleHeader>(header));
        writer.Position(end_pos);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="callable"></param>
    private void WriteCallableBinary(BinaryWriter writer, IR.Callable callable)
    {
        void writeShape(ReadOnlySpan<int> shape)
        {
            writer.Write((uint)shape.Length);
            foreach (var dim in shape)
                writer.Write((uint)dim);
        }
        var funcSched = callable.SchedResult!;

        // Skip function header
        var header_pos = writer.Position();
        writer.Skip((ulong)Marshal.SizeOf<FunctionHeader>());

        // inputs
        foreach (var input in funcSched.Inputs) { writer.Write(CodeGenUtil.StructToBytes<Schedule.MemoryRange>(input)); }
        foreach (var shape in funcSched.InputShapes) { writeShape(shape); }

        // outputs
        foreach (var output in funcSched.Outputs) { writer.Write(CodeGenUtil.StructToBytes<Schedule.MemoryRange>(output)); }
        foreach (var shape in funcSched.OutputShapes) { writeShape(shape); }

        writer.AlignPosition(8);
        var end_pos = writer.Position();

        // header
        var header = new FunctionHeader
        {
            HeaderSize = (uint)Marshal.SizeOf<FunctionHeader>(),
            Size = (uint)(end_pos - header_pos),
            InputPoolSize = (uint)funcSched.InputPoolSize,
            OutputPoolSize = (uint)funcSched.OutputPoolSize,
            Inputs = (uint)funcSched.Inputs.Count(),
            Outputs = (uint)funcSched.Outputs.Count(),
            Entrypoint = (uint)_entryPoints[callable],
            TextSize = (uint)(_functionTextEnd[callable] - _entryPoints[callable]),
        };
        writer.Position(header_pos);
        writer.Write(CodeGenUtil.StructToBytes<FunctionHeader>(header));
        writer.Position(end_pos);
    }

    /// <inheritdoc/>
    public string Dump(string name, string dumpDirPath)
    {
        if (!IsSerialized) Serialize();
        var dump_path = Path.Join(dumpDirPath, name + '.' + SourceExt);
        File.Copy(_sourcePath, dump_path);
        return dump_path;
    }

    /// <inheritdoc/>
    public ISerializeResult Serialize()
    {
        if (!IsSerialized)
        {
            _sourcePath = CodeGenUtil.GetTempFileName(SourceExt);
            using var f = new FileStream(_sourcePath, FileMode.OpenOrCreate, FileAccess.ReadWrite);
            using var bw = new BinaryWriter(f, Encoding.Default, true);
            Compile();
            Link();
            WriteBinary(bw);
            IsSerialized = true;
        }

        return new KModuleSerializeResult(Alignment);
    }

    private string _sourcePath = string.Empty;
    private readonly IR.IRModel _model;
    private readonly IR.IRModule _module;
    private IR.Callable _currentFunction;
    private readonly SortedDictionary<string, Section> _sectionWriters = new(StringComparer.CurrentCulture);
    private readonly SortedDictionary<string, RdataMergeInfo> _rdataSectionMerges = new(StringComparer.CurrentCulture);
    private readonly Dictionary<string, (ulong Offset, string Name)> _symbolOffsets;
    private readonly Dictionary<IR.Callable, ulong> _entryPoints;
    private readonly Dictionary<IR.Callable, ulong> _functionTextEnd;
}