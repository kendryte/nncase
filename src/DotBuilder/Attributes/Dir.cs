namespace DotBuilder.Attributes
{
    public class Dir : Attribute, IEdgeAttribute
    {
        public Dir(string value) : base(value)
        {
        }

        public static Dir Forward => new Dir("forward");
        public static Dir None => new Dir("none");
    }
}