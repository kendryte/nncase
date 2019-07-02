using System;

namespace DotBuilder.Attributes
{
    public class Label : Attribute, INodeAttribute, IEdgeAttribute, IGraphAttribute
    {
        public Label(string value) : base(value)
        {
        }

        [Obsolete]
        public static Label With(string label) => new Label(label);
        public static Label Set(string label) => new Label(label);
    }
}