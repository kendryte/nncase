namespace DotBuilder.Attributes
{
    public class Page : Attribute, IGraphAttribute
    {
        public Page(double width, double height) : base($"\"{width},{height}\"")
        {
        }

        public static Page A4 => new Page(11.7, 16.5);
        public static Page A3 => new Page(8.3, 11.7);

        public static Page Size(double width, double height) => new Page(width, height);
    }
}