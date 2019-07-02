namespace DotBuilder.Statements
{
    public class Graph : GraphBase
    {
        protected Graph(string graphType, string name) : base(graphType, name)
        {
        }

        public static Graph Directed(string name) => new Graph("digraph", name);
        public static Graph UnDirected(string name) => new Graph("graph", name);
    }
}