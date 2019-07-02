namespace DotBuilder.Attributes
{
    public class Font : Attribute, INodeAttribute, IEdgeAttribute, IGraphAttribute
    {
        public Font(string name, string value) : base(name, value)
        {
        }

        public static Font Size(double size) => new Font("fontsize", $"{size}");
#pragma warning disable CS0108 // Member hides inherited member; missing new keyword
        public static Font Name(string name) => new Font("fontname", name);
#pragma warning restore CS0108 // Member hides inherited member; missing new keyword
    }
}