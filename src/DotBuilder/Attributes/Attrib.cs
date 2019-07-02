namespace DotBuilder.Attributes
{
    public class Attrib : Attribute, INodeAttribute, IEdgeAttribute, IGraphAttribute
    {
        public Attrib(string name, string value) : base(name, value)
        {
        }

        public static Attrib Set(string name, string value) => new Attrib(name,value);
    }
}