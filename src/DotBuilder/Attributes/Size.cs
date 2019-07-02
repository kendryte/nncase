namespace DotBuilder.Attributes
{
    public class Size : Attribute, IGraphAttribute
    {
        public Size(double width, double height) : base($"\"{width},{height}\"")
        {
        }

        public static Size A4 => new Size(11.7, 16.5);
        public static Size A3 => new Size(8.3, 11.7);

        public static Size SizeMM(double width, double height) => new Size(width/25.4, height/25.4);
        public static Size SizeIN(double width, double height) => new Size(width, height);
    }
}