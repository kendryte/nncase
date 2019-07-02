namespace DotBuilder.Statements
{
    public class Subgraph : GraphBase, IStatement
    {
        private static int _instance;

        protected Subgraph(string graphType, string name) : base(graphType, name)
        {
        }

        public static Subgraph Cluster => new Subgraph("subgraph", "cluster_" + _instance++);
    }
}