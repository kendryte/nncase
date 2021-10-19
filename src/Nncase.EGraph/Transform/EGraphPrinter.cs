using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Transform;
using GiGraph.Dot.Entities.Graphs;
using GiGraph.Dot.Entities.Nodes;
using GiGraph.Dot.Entities.Clusters;
using GiGraph.Dot.Extensions;
using GiGraph.Dot.Types.Graphs;
using GiGraph.Dot.Types.Nodes;
using GiGraph.Dot.Types.Styling;
using GiGraph.Dot.Types.Colors;
using GiGraph.Dot.Types.Records;
using GiGraph.Dot.Types.Edges;
using Nncase.IR.Math;
using System.Drawing;

namespace Nncase.Transform
{

    public class EGraphPrinter
    {

        private readonly Dictionary<EClass, DotCluster> _classes = new Dictionary<EClass, DotCluster>();

        public DotGraph ConvertEGraphAsDot(EGraph eGraph)
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
                _classes.Add(eclass, eclassCluster);

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
            return g;
        }
        public DotGraph ConvertEGraphAsDot(EGraph eGraph, List<(EClass, Dictionary<string, EClass>)> matches)
        {
            DotGraph g = ConvertEGraphAsDot(eGraph);
            Array knowcolors = Enum.GetValues(typeof(KnownColor));
            var random = new Random(123);
            int count = 0;
            foreach (var (parentEclass, env) in matches)
            {
                var eclassCluster = _classes[parentEclass];
                Color color = (Color)knowcolors.GetValue(random.Next(knowcolors.Length - 1));
                eclassCluster.Nodes.Add($"m{eclassCluster.Id}_{count}", node =>
                {
                    node.Label = $"^m{count}";
                    node.Color = color;
                    node.Shape = DotNodeShape.Circle;
                    node.Size.Height = 2;
                    node.Size.Width = 2;
                });
                foreach (var (name, childEclass) in env)
                {
                    var childeclassCluster = _classes[childEclass];
                    childeclassCluster.Nodes.Add($"m{childEclass.Id}_{count}", node =>
                    {
                        node.Label = $">m{count}";
                        node.Color = color;
                        node.Shape = DotNodeShape.Circle;
                        node.Size.Height = 2;
                        node.Size.Width = 2;
                    });
                }
            }
            return g;
        }

        public DotGraph SaveToFile(DotGraph g, string file)
        {
            g.Build();
            g.SaveToFile(file);
            return g;
        }

        public static DotGraph DumpEgraphAsDot(EGraph eGraph, string file)
        {
            return DumpEgraphAsDot(eGraph, new List<(EClass, Dictionary<string, EClass>)> { }, file);
        }

        public static DotGraph DumpEgraphAsDot(EGraph eGraph, List<(EClass, Dictionary<string, EClass>)> matches, string file)
        {
            var printer = new EGraphPrinter();
            var g = printer.ConvertEGraphAsDot(eGraph, matches);
            return printer.SaveToFile(g, file);
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
