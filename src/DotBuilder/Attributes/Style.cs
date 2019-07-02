namespace DotBuilder.Attributes
{
    public class Style : Attribute, INodeAttribute, IGraphAttribute, IEdgeAttribute, IMustBeCombined
    {
        private Style(string value) : base(value)
        {
        }

        public static Style Solid => new Style("solid");
        public static Style Dashed => new Style("dashed");
        public static Style Dotted => new Style("dotted");
        public static Style Bold => new Style("bold");
        public static Style Rounded => new Style("rounded");
        public static Style Diagonals => new Style("diagonals");
        public static Style Filled => new Style("filled");
        public static Style Striped => new Style("striped");
    }
}