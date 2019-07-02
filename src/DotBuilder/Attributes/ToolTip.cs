namespace DotBuilder.Attributes
{
    public abstract class ToolTipBase : Attribute
    {
        protected ToolTipBase(string value) : base(ConvertEscapeCharacters(value))
        {
        }

        private static string ConvertEscapeCharacters(string value)
        {
            value = value.Replace("\n", "&#13;");
            value = value.Replace("\t", "&#9;");

            return value;
        }
    }

    public class ToolTip : ToolTipBase, INodeAttribute, IEdgeAttribute
    {
        public ToolTip(string value) : base(value)
        {
        }

        public static ToolTip Set(string value) => new ToolTip(value);
    }

    public class LabelToolTip : ToolTipBase, IEdgeAttribute
    {
        public LabelToolTip(string value) : base(value)
        {
        }

        public static LabelToolTip Set(string value) => new LabelToolTip(value);
    }
}