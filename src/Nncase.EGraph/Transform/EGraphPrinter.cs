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
            foreach (var eclass in eGraph.Classes)
            {
                var eclassNode = new DotNode($"{eclass.Id}")
                {
                    Label = $"{eclass.Id}",
                    Shape = DotNodeShape.Circle
                };
                g.Nodes.Add(eclassNode);

                foreach (var enode in eclass.Nodes)
                {
                    string exprId = enode.Expr.GetHashCode().ToString();

                    var args = new List<DotRecordTextField> {
                      new DotRecordTextField(visitor.Visit(enode.Expr),
                                             "Type") };

                    for (int i = 0; i < enode.Children.Length; i++)
                    {
                        args.Add(new DotRecordTextField(null, $"P{i}"));
                    }
                    var exprNode = g.Nodes.Add(exprId);
                    exprNode.ToRecordNode(new DotRecord(args));
                    for (int i = 0; i < enode.Children.Length; i++)
                    {
                        g.Edges.Add($"{enode.Children[i].Id}", exprNode, edge =>
                       {
                           edge.Head.Endpoint.Port = new DotEndpointPort($"P{i}");
                       });
                    }

                    // edge eclass with enode
                    g.Edges.Add(eclassNode, exprNode, edge =>
                    {
                        edge.Style.LineStyle = DotLineStyle.Dashed;
                    });
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
                        name += " " + (expr.Data.ToString());
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
