using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Transform;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Styling;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Edges;
using Nncase.IR.Math;

namespace Nncase.Transform
{
    public static class EGraphPrinter
    {
        public static DotGraph DumpEgraphAsDot(EGraph eGraph, string file)
        {
            DotDumpVisitor visitor = new DotDumpVisitor();
            var g = new DotGraph(directed: true);
            g.Clusters.AllowEdgeClipping = true;

            foreach (var eclass in eGraph.Classes.Select(x => x.Find()).Distinct())
            {
                // make eclass as cluster
                var eclassCluster = g.Clusters.Add($"{eclass.Id}", cluster =>
               {
                   cluster.Style.BorderStyle = DotBorderStyle.Dotted;
                   cluster.Label = $"{eclass.Id}";
               });

                eclassCluster.Nodes.Add(new DotNode(eclassCluster.Id + "dummy"), node =>
                  {
                      node.Label = "";
                      node.Style.Invisible = true;
                      node.Size.Height = 0;
                      node.Size.Width = 0;
                  });

                foreach (var enode in eclass.Nodes)
                {
                    string exprId = enode.Expr.GetHashCode().ToString();

                    var args = new List<DotRecordTextField> {
                      new DotRecordTextField(visitor.Visit(enode.Expr), "Type") };

                    for (int i = 0; i < enode.Children.Count; i++)
                    {
                        args.Add(new DotRecordTextField(null, $"P{i}"));
                    }

                    var exprNode = eclassCluster.Nodes.Add(exprId);

                    exprNode.ToRecordNode(new DotRecord(args));
                    for (int i = 0; i < enode.Children.Count; i++)
                    {
                        // var pnode =  from pnode in select
                        g.Edges.Add($"{enode.Children[i].Find().Id}" + "dummy", exprNode, edge =>
                         {
                             edge.Tail.ClusterId = $"{enode.Children[i].Id}";
                             edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                         });
                    }
                }
            }
            g.Build();
            g.SaveToFile(file);
            return g;
        }

        private class DotDumpVisitor : ExprFunctor<string, string>
        {
            public override string Visit(Call expr)
            {
                return expr.GetType().Name;
            }

            public override string Visit(Const expr)
            {

                string name = expr.GetType().Name;
                if (expr.CheckedType is not null)
                {
                    return name;
                }
                if (expr.ValueType is TensorType)
                {
                    if (((TensorType)expr.ValueType).IsScalar)
                    {
                        object data = ((TensorType)expr.ValueType).DataType switch
                        {
                            DataType.Bool => BitConverter.ToBoolean(expr.Data),
                            DataType.Int16 => BitConverter.ToInt16(expr.Data),
                            DataType.Int32 => BitConverter.ToInt32(expr.Data),
                            DataType.Int64 => BitConverter.ToInt64(expr.Data),
                            DataType.UInt16 => BitConverter.ToUInt16(expr.Data),
                            DataType.UInt32 => BitConverter.ToUInt32(expr.Data),
                            DataType.UInt64 => BitConverter.ToUInt64(expr.Data),
                            DataType.Float64 => BitConverter.ToDouble(expr.Data),
                            _ => "InVaild"
                        };
                        name += " " + data.ToString();
                    }
                }
                return name;
            }

            public override string Visit(Function expr)
            {
                return expr.GetType().Name;
            }

            public override string Visit(Op expr)
            {
                return expr switch
                {
                    Unary op => op.UnaryOp.ToString(),
                    Binary op => op.BinaryOp.ToString(),
                    _ => expr.GetType().Name,
                };
            }

            public override string Visit(Var expr)
            {

                return expr.GetType().Name + " " + expr.Name;
            }

            public override string VisitType(AnyType type) => "any";

            public override string VisitType(CallableType type) =>
                $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

            public override string VisitType(InvalidType type) => "invalid";

            public override string VisitType(TensorType type) =>
                $"{DataTypes.GetDisplayName(type.DataType)}{type.Shape}";

            public override string VisitType(TupleType type) =>
                $"({string.Join(", ", type.Fields.Select(VisitType))})";

        }
    }
}
