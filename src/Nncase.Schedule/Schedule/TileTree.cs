// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Affine;
using VisitorPatternGenerator;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule;

public partial interface ITreeNode
{
    public ITreeNode? Parent { get; set; }
}

public interface ITileAbleNode
{
    int Level { get; }

    int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    string[] Vars { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    Isl.basic_map DomainRelation { get; set; }
}

[Acceptor<ITreeNode, ScopeNode>]
public partial class ScopeNode
{
    private readonly List<ITreeNode> _children;

    public ScopeNode(ITreeNode? parent = null)
    {
        Parent = parent;
        _children = new();
    }

    public ITreeNode? Parent { get; set; }

    public IList<ITreeNode> Children => _children;

    public void Add(ITreeNode node)
    {
        node.Parent = this;
        _children.Add(node);
    }

    public void Insert(int index, ITreeNode node)
    {
        node.Parent = this;
        _children.Insert(index, node);
    }

    public void InsertRange(int index, IList<ITreeNode> nodes)
    {
        foreach (var item in nodes)
        {
            item.Parent = this;
        }

        _children.InsertRange(index, nodes);
    }

    public void Remove(ITreeNode node)
    {
        _children.Remove(node);
        node.Parent = null;
    }
}

[Acceptor<ITreeNode, TileNode>]
public partial class TileNode : ITileAbleNode
{
    private ITreeNode _child;

    public TileNode(int level, int opId, string[] vars)
    {
        Level = level;
        OpId = opId;
        Vars = vars;
        DomainRelation = TilingUtilities.GetIdentityMap(vars.Length, $"op{OpId}", $"op{OpId}");
        _child = null!;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public string[] Vars { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public Isl.basic_map DomainRelation { get; set; }

    public ITreeNode Child
    {
        get => _child; set
        {
            _child = value;
            _child.Parent = this;
        }
    }
}

[Acceptor<ITreeNode, OpNode>]
public partial class OpNode : ITileAbleNode
{
    public OpNode(int opId, string[] vars, int[] domain, int[][] bufferShapes, Isl.basic_map[] reads, Isl.basic_map write, Dependence[] dependences)
    {
        Level = 0;
        OpId = opId;
        Vars = vars;
        DomainRelation = TilingUtilities.GetIdentityMap(vars.Length, $"op{OpId}", $"op{OpId}");
        DomainBounds = domain;
        BufferShapes = bufferShapes;
        Reads = reads;
        Write = write;
        Dependences = dependences;
    }

    public ITreeNode? Parent { get; set; }

    public int Level { get; }

    public int OpId { get; }

    /// <summary>
    /// Gets the domain var names.
    /// </summary>
    public string[] Vars { get; }

    /// <summary>
    /// Gets or sets the domain relation which from parent domain map to current node's domain.
    /// </summary>
    public Isl.basic_map DomainRelation { get; set; }

    public IReadOnlyList<Dependence> Dependences { get; }

    public IReadOnlyList<int> DomainBounds { get; }

    public int[][] BufferShapes { get; }

    public IReadOnlyList<Isl.basic_map> Reads { get; }

    public Isl.basic_map Write { get; }

    public record Dependence(int Index, OpNode Node)
    {
    }
}

[Visitor<ITreeNode>(true)]
public partial interface ITreeNodeVisitor<in TArg1> { }

public record TileTreePrinterContext(int ParentOpId, IReadOnlyList<string> Names, int Indent)
{
    public static TileTreePrinterContext Default => new(-1, Array.Empty<string>(), 0);
}

public class TileTreePrinter : ITreeNodeVisitor<TileTreePrinterContext>
{
    private StreamWriter _writer;

    public TileTreePrinter(StreamWriter writer)
    {
        _writer = writer;
    }

    public void Indent(int indent)
    {
        var s = new string(' ', indent);
        _writer.Write(s);
    }

    public void Visit(ScopeNode value, TileTreePrinterContext context)
    {
        var indent = context.Indent;
        Indent(indent);
        _writer.WriteLine($"# scope");
        foreach (var child in value.Children)
        {
            child.Accept(this, context with { Indent = indent });
        }
    }

    public void Visit(TileNode value, TileTreePrinterContext context)
    {
        var (pid, pnames, indent) = context;
        Indent(indent);
        _writer.WriteLine($"# Tile Op {value.OpId} at level {value.Level}");

        var names = MappingVars(value, pid, pnames);
        Indent(indent);
        var ivs = string.Join(",", Enumerable.Range(0, names.Length).Select(i => $"i{i}"));
        _writer.WriteLine($"for ({ivs}) in range({string.Join(", ", names)}):");
        indent++;

        value.Child?.Accept(this, context with { ParentOpId = value.OpId, Names = names, Indent = indent });
    }

    public void Visit(OpNode value, TileTreePrinterContext context)
    {
        var (pid, pnames, indent) = context;
        var names = MappingVars(value, pid, pnames);
        Indent(indent);
        _writer.WriteLine($"# Compute Op {value.OpId} at level {value.Level}");

        Indent(indent);
        var ivs = string.Join(", ", Enumerable.Range(0, names.Length).Select(i => $"d{i}"));
        _writer.WriteLine($"with ({string.Join(", ", names)}) as ({ivs}):");
        indent++;

        var set = value.Dependences.ToDictionary(d => d.Index, d => d.Node);
        for (int i = 0; i < value.Reads.Count; i++)
        {
            Indent(indent);

            _writer.Write($"read_{i} = ");
            _writer.Write(value.Reads[i]);
            if (set.ContainsKey(i))
            {
                _writer.WriteLine($" @ Op{set[i].OpId}");
            }
            else
            {
                _writer.WriteLine();
            }
        }

        Indent(indent);
        _writer.Write("write = ");
        _writer.WriteLine(value.Write);
    }

    private string[] MappingVars(ITileAbleNode value, int parentId, IReadOnlyList<string> pnames)
    {
        var names = value.Vars.ToArray();

        if (pnames.Any())
        {
            var space = Isl.space.unit(Isl.ctx.Instance);
            space = space.add_named_tuple(new Isl.id(Isl.ctx.Instance, $"op{parentId}"), (uint)pnames.Count);
            var set = Isl.set.universe(space);
            for (int i = 0; i < pnames.Count; i++)
            {
                set = set.fix_dim_si((uint)i, i);
            }

            var res = set.apply(value.DomainRelation);
            for (int i = 0; i < value.DomainRelation.dim(Isl.dim_type.out_); i++)
            {
                var val = res.dim_max_val(i);
                if (val.is_int())
                {
                    names[i] = pnames[checked((int)val.num_si())];
                }
            }
        }

        return names;
    }
}

public partial class TileTreeMerger : ITreeNodeVisitor<Unit>
{
    public TileTreeMerger(int opConsumer, int opProducer, int level)
    {
        OpConsumer = opConsumer;
        OpProducer = opProducer;
        Level = level;
    }

    public int OpConsumer { get; }

    public int OpProducer { get; }

    public int Level { get; }

    public void Visit(ScopeNode value, Unit arg1)
    {
        for (int i = 0; i < value.Children.Count - 1; i++)
        {
            if (value.Children[i] is TileNode producer && value.Children[i + 1] is TileNode consumer &&
                producer.Level == Level && consumer.Level == Level &&
                producer.OpId == OpProducer && consumer.OpId == OpConsumer)
            {
                if (PerformMerge(value, consumer, producer))
                {
                    return;
                }
            }
        }

        foreach (var item in value.Children)
        {
            item.Accept(this, arg1);
        }
    }

    public void Visit(TileNode value, Unit arg1)
    {
        value.Child.Accept(this, arg1);
    }

    public void Visit(OpNode value, Unit arg1)
    {
    }

    private bool FindOpNode(ITreeNode treeNode, int opId, out OpNode retNode)
    {
        switch (treeNode)
        {
            case OpNode opNode:
                if (opNode.OpId == opId)
                {
                    retNode = opNode;
                    return true;
                }

                break;

            case TileNode tNode:
                return FindOpNode(tNode.Child, opId, out retNode);
            case ScopeNode sNode:
                for (int i = 0; i < sNode.Children.Count; i++)
                {
                    if (FindOpNode(sNode.Children[i], opId, out retNode))
                    {
                        return true;
                    }
                }

                break;
        }

        retNode = null!;
        return false;
    }

    private bool FindFristOpNode(ITreeNode treeNode, out OpNode retNode)
    {
        switch (treeNode)
        {
            case OpNode opNode:
                retNode = opNode;
                return true;
            case TileNode tNode:
                return FindFristOpNode(tNode.Child, out retNode);
            case ScopeNode sNode:
                for (int i = 0; i < sNode.Children.Count; i++)
                {
                    if (FindFristOpNode(sNode.Children[i], out retNode))
                    {
                        return true;
                    }
                }

                break;
        }

        retNode = null!;
        return false;
    }

    private Isl.basic_map CheckFullMapping(Isl.basic_map domainRel, OpNode writeOp)
    {
        // domainRel = readAccess.apply_range(writeAccess.reverse()); // the read dependence the write
        var eqMat = domainRel.equalities_matrix(Isl.dim_type.in_, Isl.dim_type.out_, Isl.dim_type.cst, Isl.dim_type.param, Isl.dim_type.div);

        var domainVarMap = new Dictionary<int, int>();
        for (int r = 0; r < eqMat.rows(); r++)
        {
            bool noCoff = true;
            for (int i = domainRel.dim(Isl.dim_type.in_) + domainRel.dim(Isl.dim_type.out_); i < eqMat.cols(); i++)
            {
                noCoff &= eqMat.element_val(r, i).is_zero();
            }

            if (!noCoff)
            {
                continue;
            }

            for (int i = 0; i < domainRel.dim(Isl.dim_type.in_); i++)
            {
                var inv = eqMat.element_val(r, i);
                for (int j = 0; j < domainRel.dim(Isl.dim_type.out_); j++)
                {
                    var outv = eqMat.element_val(r, j + domainRel.dim(Isl.dim_type.in_));
                    if (!inv.is_zero() && !outv.is_zero() && inv.add(outv).is_zero())
                    {
                        if (!domainVarMap.TryGetValue(i, out _))
                        {
                            domainVarMap.Add(i, j);
                        }
                        else
                        {
                            throw new InvalidOperationException("the same input dim can't equal to muli output dim");
                        }
                    }
                }
            }
        }

        // rebuild the domain relation
        var space = domainRel.space();
        var ls = Isl.local_space.from_space(space);
        var uniMap = Isl.basic_map.universe(space);
        foreach (var k in domainVarMap.Keys)
        {
            var cons = Isl.constraint.alloc_equality(ls);
            cons = cons.set_coefficient_si(Isl.dim_type.in_, k, -1);
            cons = cons.set_coefficient_si(Isl.dim_type.out_, domainVarMap[k], 1);
            uniMap = uniMap.add_constraint(cons);
        }

        return uniMap;
    }

    private bool PerformMerge(ScopeNode parent, TileNode consumer, TileNode producer)
    {
        if (!FindFristOpNode(consumer, out var firstConsumerOp))
        {
            return false;
        }

        if (firstConsumerOp.Dependences.Count != 1)
        {
            return false;
        }

        // 1. compute the domain realtion : first_consumer_op domain -> producer domain
        var writeOp = firstConsumerOp.Dependences[0].Node;
        var readAccess = firstConsumerOp.Reads[firstConsumerOp.Dependences[0].Index];
        Isl.basic_map domainRel = readAccess.apply_range(writeOp.Write.reverse());

        // 2. check the domain rel
        domainRel = CheckFullMapping(domainRel, writeOp);

        // 3. compose with merged consumer op's domain realtion.
        if (consumer.Child is ScopeNode subConsumerScope)
        {
            domainRel = subConsumerScope.Children.OfType<ITileAbleNode>().First().DomainRelation.apply_range(domainRel);
        }

        // 4. modify the tree.
        parent.Remove(producer);
        var nextLevelProducer = producer.Child;
        if (consumer.Child is ScopeNode subScope)
        {
            AddProducerToScope(subScope, nextLevelProducer);
        }
        else
        {
            subScope = new ScopeNode();
            AddProducerToScope(subScope, nextLevelProducer);
            subScope.Add(consumer.Child);
            consumer.Child = subScope;
        }

        // when the 
        if (nextLevelProducer is ScopeNode producerScope)
        {
            // tileAble.DomainRelation = domainRel;
            foreach (var tnode in producerScope.Children.OfType<ITileAbleNode>())
            {
                tnode.DomainRelation = domainRel.apply_range(tnode.DomainRelation);
            }
        }
        else if (nextLevelProducer is ITileAbleNode tileAble)
        {
            tileAble.DomainRelation = domainRel;
        }

        return true;
    }

    private void AddProducerToScope(ScopeNode scopeNode, ITreeNode producer)
    {
        if (producer is ScopeNode nextLevelProduceScope)
        {
            scopeNode.InsertRange(0, nextLevelProduceScope.Children);
        }
        else
        {
            scopeNode.Insert(0, producer);
        }
    }
}

public static class TreeSearch
{
    public static OpNode BuildTree(Grid current, ScopeNode scope, int level, ref int opId)
    {
        var dependences = new List<OpNode.Dependence>();
        for (int i = 0; i < current.Reads.Length; i++)
        {
            if (current.Reads[i] is Grid producer)
            {
                var producerNode = BuildTree(producer, scope, level, ref opId);
                dependences.Add(new OpNode.Dependence(i, producerNode));
                opId++;
            }
        }

        var bufferShapes = current.Buffers.AsValueEnumerable().Select(TilingUtilities.GetBufferShape).ToArray();
        var domain = TilingUtilities.InferDomainBounds(bufferShapes, current.AccessMaps.ToArray());
        var copId = opId;
        var domainDims = current.AccessMaps[0].Domains.Length;
        var vars = Enumerable.Range(0, domainDims).Select(i => $"op{copId}_d{i}").ToArray();
        var readAccess = new Isl.basic_map[current.AccessMaps.Length - 1];
        for (int i = 0; i < readAccess.Length; i++)
        {
            readAccess[i] = current.AccessMaps[i].AsIslMap($"op{copId}", domain);
        }

        var opNode = new OpNode(copId, vars, domain, bufferShapes, readAccess, current.AccessMaps[^1].AsIslMap($"op{copId}", domain), dependences.ToArray());
        var tileNodeRoot = new TileNode(level, copId, vars);
        TileNode tileNodeTail = tileNodeRoot;
        for (int l = level - 1; l >= 1; l--)
        {
            var child = new TileNode(l, copId, vars);
            tileNodeTail.Child = child;
            tileNodeTail = child;
        }

        tileNodeTail.Child = opNode;
        scope.Add(tileNodeRoot);
        return opNode;
    }

    public static void Dump(ITreeNode tree, string name)
    {
        using (var stream = Diagnostics.DumpScope.Current.OpenFile($"{name}.py"))
        {
            using var writer = new StreamWriter(stream);
            var printer = new TileTreePrinter(writer);
            tree.Accept(printer, TileTreePrinterContext.Default);
            writer.Flush();
        }
    }

    public static void Merge(ITreeNode tree, int opConsumer, int opProducer, int level)
    {
        var merger = new TileTreeMerger(opConsumer, opProducer, level);
        tree.Accept(merger, default);
    }

    public static void Search(Grid grid)
    {
        var tree = new ScopeNode();
        var opId = 0;
        BuildTree(grid, tree, 2, ref opId);
        Dump(tree, "build");

        // try merge op2 and op1 at level 1
        Merge(tree, 2, 1, 2);
        Dump(tree, "merge_2_1_2");

        Merge(tree, 2, 0, 2);
        Dump(tree, "merge_2_0_2");

        // merge 1 0 1
        Merge(tree, 1, 0, 1);
        Dump(tree, "merge_1_0_1");

        Merge(tree, 2, 1, 1);
        Dump(tree, "merge_2_1_1");

        // first find the producer comsumer struct
        // grid.Buffers
        // three op
        // matmul
        // exp
        // matmul

        // tree.Accept()
        // tree.Add(new TileNode());
        // tree.Add(new TileNode());
        // tree.Add(new TileNode());
    }
}
