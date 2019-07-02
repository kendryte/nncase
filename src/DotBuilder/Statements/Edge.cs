using DotBuilder.Attributes;

namespace DotBuilder.Statements
{
    public class Edge : Statement<Edge, IEdgeAttribute>
    {
        private readonly string _from;
        private readonly string _to;

        private Edge(string from, string to)
        {
            _from = from;
            _to = to;
        }

        public override string Render()
        {
            return $"\"{_from}\"->\"{_to}\" {base.Render()}";
        }

        public static Edge Between(string from, string to) => new Edge(from, to);
    }
}