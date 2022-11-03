using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR.Builders;

/// <summary>
/// builfer the block.
/// </summary>
public interface IBlockBuilder : IExprBuilder<Block>
{
    /// <summary>
    /// else block.
    /// </summary>
    /// <param name="exprOrBuilders"> statements. </param>
    /// <returns> BlockBuilder. </returns>
    IBlockBuilder Body(params object[] exprOrBuilders);

    /// <summary>
    /// then block.
    /// </summary>
    /// <param name="exprOrBuilders"> statements. </param>
    /// <returns> BlockBuilder. </returns>
    IBlockBuilder Init(params object[] exprOrBuilders);

    /// <summary>
    /// create the iterVar and bind the value.
    /// </summary>
    /// <param name="vi"></param>
    /// <param name="domain"></param>
    /// <param name="mode"></param>
    /// <param name="value"></param>
    /// <returns></returns>
    IBlockBuilder Bind(out IterVar vi, Range domain, IterationMode mode, Var value);

    /// <summary>
    /// bind the itervar with for loop.
    /// </summary>
    /// <param name="vi"></param>
    /// <param name="fi"></param>
    /// <param name="iterType"></param>
    /// <returns></returns>
    /// <exception cref="NotSupportedException"></exception>
    IBlockBuilder Remap(out IterVar vi, For fi, char iterType);

    /// <summary>
    /// alloctions
    /// </summary>
    /// <param name="buffers"></param>
    /// <returns></returns>
    IBlockBuilder Alloc(params object[] buffers);

    /// <summary>
    /// reads
    /// </summary>
    /// <param name="buffer_regions"></param>
    /// <returns></returns>
    IBlockBuilder Reads(params object[] buffer_regions);

    /// <summary>
    /// writes
    /// </summary>
    /// <param name="buffer_regions"></param>
    /// <returns></returns>
    IBlockBuilder Writes(params object[] buffer_regions);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="predicate"></param>
    /// <returns></returns>
    IBlockBuilder Predicate(Expr predicate);
}

internal class BlockBuilder : IBlockBuilder
{
    private readonly string _name;
    private readonly List<object> _init = new();
    private readonly List<object> _body = new();
    private readonly List<IterVar> _iterVars = new();
    private readonly List<TIR.Buffer> _allocations = new();
    private readonly List<TIR.BufferRegion> _reads = new();
    private readonly List<TIR.BufferRegion> _writes = new();
    private Expr? _predicate;

    public BlockBuilder(string name)
    {
        _name = name;
    }

    public IBlockBuilder Body(params object[] exprOrBuilders)
    {
        _body.AddRange(exprOrBuilders);
        return this;
    }

    public IBlockBuilder Init(params object[] exprOrBuilders)
    {
        _init.AddRange(exprOrBuilders);
        return this;
    }

    public IBlockBuilder Bind(out IterVar vi, Range dom, IterationMode mode, Var value)
    {
        vi = new IterVar(dom, mode, value);
        _iterVars.Add(vi);
        return this;
    }

    public IBlockBuilder Remap(out IterVar vi, For fi, char iterType)
    {
        var toMode = (char x) => x switch
        {
            'S' => IterationMode.DataParallel,
            'R' => IterationMode.CommReduce,
            _ => throw new NotSupportedException("Only Support \"S\" (for Spatial) or \"R\" ( Reduce)"),
        };
        return Bind(out vi, fi.Domain, toMode(iterType), fi.LoopVar);
    }

    public Block Build()
    {
        return new(_name, Sequential.Flatten(_body), Sequential.Flatten(_init), new(_iterVars), new(_reads), new(_writes), new(_allocations), _predicate ?? true);
    }

    private static void Add<T>(List<T> list, object[] inputs)
    {
        foreach (var obj in inputs)
        {
            switch (obj)
            {
                case T item:
                    list.Add(item);
                    break;
                case IEnumerable<T> items:
                    list.AddRange(items.OfType<T>());
                    break;
                default:
                    break;
            }
        }
    }

    public IBlockBuilder Alloc(params object[] buffers)
    {
        Add(_allocations, buffers);
        return this;
    }

    public IBlockBuilder Reads(params object[] buffer_regions)
    {
        Add(_reads, buffer_regions);
        return this;
    }

    public IBlockBuilder Writes(params object[] buffer_regions)
    {
        Add(_writes, buffer_regions);
        return this;
    }

    public IBlockBuilder Predicate(Expr predicate)
    {
        _predicate ??= predicate;
        return this;
    }
}
