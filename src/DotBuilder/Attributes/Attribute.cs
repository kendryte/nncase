namespace DotBuilder.Attributes
{
    public interface IAttribute
    {
        string Name { get; }
        string Value { get; }
        string Render();
    }

    public interface IGraphAttribute : IAttribute
    {
    }

    public interface INodeAttribute : IAttribute
    {
    }

    public interface IEdgeAttribute : IAttribute
    {
    }

    public interface ISubgraphAttribute : IAttribute
    {
    }


    /// <summary>
    /// Indicated that an attribute must be combined into a single item. E.g. style="rounded,filled"
    /// </summary>
    public interface IMustBeCombined
    {
        
    }

    public abstract class Attribute : IAttribute
    {
        public string Name { get; }
        public string Value { get; }
        protected Attribute(string value)
        {
            Value = value;
            Name = GetType().Name.ToLower();
        }
        protected Attribute(string name, string value)
        {
            Name = name;
            Value = value;
        }

        public virtual string Render()
        {
            return $"{Name}=\"{Value}\"";
        }
    }
}