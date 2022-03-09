// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Edges;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Styling;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Transform;

namespace Nncase.Transform;

public partial class EGraphPrinter
{
    private readonly Dictionary<EClass, DotCluster> ClusterMaps = new Dictionary<EClass, DotCluster>();

    private readonly Dictionary<EClass, string> OpMaps = new();

    private readonly EGraph eGraph;

    private readonly DotDumpVisitor visitor = new DotDumpVisitor();

    public readonly DotGraph dotGraph;

    public EGraphPrinter(EGraph _eGraph)
    {
        dotGraph = new(directed: true);
        dotGraph.Clusters.AllowEdgeClipping = true;
        eGraph = _eGraph;
        foreach (var eclass in eGraph.Classes)
        {
            if (eclass.Nodes.Count == 1 && eclass.Nodes[0].Expr is Op op)
            {
                if (!OpMaps.ContainsKey(eclass))
                {
                    var name = visitor.Visit(op);
                    OpMaps.Add(eclass, name);
                }
            }
        }
    }

    public DotGraph ConvertEGraphAsDot()
    {
        foreach (var eClass in eGraph.Classes.Where(x => !OpMaps.ContainsKey(x)))
        {
            // make eClass as cluster
            var eclassCluster = dotGraph.Clusters.Add($"{eClass.Id}", cluster =>
           {
               cluster.Style.BorderStyle = DotBorderStyle.Dotted;
               cluster.Label = $"{eClass.Id}";
               cluster.LabelAlignment.Horizontal = GiGraph.Dot.Types.Alignment.DotHorizontalAlignment.Left;
           });
            ClusterMaps.Add(eClass, eclassCluster);

            eclassCluster.Nodes.Add(new DotNode(eclassCluster.Id + "dummy"), node =>
              {
                  node.Label = "";
                  node.Style.Invisible = true;
                  node.Size.Height = 0;
                  node.Size.Width = 0;
              });

            foreach (var enode in eClass.Nodes)
            {
                string exprId = enode.Expr.GetHashCode().ToString();

                var args = new List<DotRecordTextField> {
                      new DotRecordTextField(visitor.Visit(enode.Expr), "Type"), };

                foreach (var (child, i) in enode.Children.Select((c, i) => (c, i)))
                {
                    var label = $"{child.Find().Id}";
                    if (OpMaps.ContainsKey(child))
                    {
                        label = OpMaps[child];
                    }

                    args.Add(new DotRecordTextField(label, $"P{i}"));
                }

                var exprNode = eclassCluster.Nodes.Add(exprId);

                // display the output type
                if (enode.Expr is Call or Function or Var)
                {
                    exprNode.ToRecordNode(rb =>
                    {
                        rb.AppendFlippedRecord(new DotRecord(args)).AppendFlippedRecord(enode.Expr.CheckedType?.DumpTypeAsIL() ?? "None");
                    }, true);
                }
                else
                {
                    exprNode.ToRecordNode(new DotRecord(args));
                }

                for (int i = 0; i < enode.Children.Count; i++)
                {
                    if (OpMaps.ContainsKey(enode.Children[i]))
                    {
                        continue;
                    }

                    // var pnode =  from pnode in select
                    dotGraph.Edges.Add($"{enode.Children[i].Find().Id}" + "dummy", exprNode, edge =>
                     {
                         edge.Tail.ClusterId = $"{enode.Children[i].Id}";
                         edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                     });
                }
            }
        }

        return dotGraph;
    }

    public DotGraph SaveToFile(string file)
    {
        if (!file.EndsWith(".dot"))
        {
            file += ".dot";
        }

        var dirName = Path.GetDirectoryName(file);
        if (dirName is not null && dirName != "")
        {
            Directory.CreateDirectory(dirName);
        }

        dotGraph.Build();
        dotGraph.SaveToFile(file);
        return dotGraph;
    }

    public static DotGraph DumpEgraphAsDot(EGraph eGraph, string file)
    {
        var printer = new EGraphPrinter(eGraph);
        printer.ConvertEGraphAsDot();
        return printer.SaveToFile(file);
    }

    private class DotDumpVisitor : ExprFunctor<string, string>
    {
        public override string Visit(Call expr)
        {
            return expr.GetType().Name;
        }

        public override string Visit(Const expr) => expr.ToString();

        public override string Visit(Function expr) => expr.GetType().Name;

        public override string Visit(Op expr)
        {
            return expr switch
            {
                Unary op => op.UnaryOp.ToString(),
                Binary op => op.BinaryOp.ToString(),
                Reduce op => "Reduce" + op.ReduceOp.ToString(),
                _ => expr.GetType().Name,
            };
        }

        public override string Visit(Var expr) => expr.GetType().Name + " " + expr.Name;
    }
}
