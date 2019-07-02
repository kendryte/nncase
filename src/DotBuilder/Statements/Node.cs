using DotBuilder.Attributes;

namespace DotBuilder.Statements
{
    public class Node : Statement<Node, INodeAttribute>
    {
        private readonly string _name;

        private Node(string name)
        {
            _name = name;
        }

        public static Node Name(string name) => new Node(name);

        public override string Render()
        {
            return $"\"{_name}\" {base.Render()}";
        }
    }
}