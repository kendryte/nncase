using System;

namespace DotBuilder.Attributes
{
    public class Url : Attribute, INodeAttribute, IEdgeAttribute, IGraphAttribute
    {
        private Url(string value) : base("URL", value)
        {
        }

        [Obsolete]
        public static Url Of(string url) => new Url(url);
        public static Url Set(string url) => new Url(url);
    }
}