namespace DotBuilder.Attributes
{
    public class FixedSize : Attribute, INodeAttribute
    {
        private FixedSize(string value) : base(value)
        {
        }

        public static FixedSize False => new FixedSize("false");
        public static FixedSize True => new FixedSize("true");
    }
}