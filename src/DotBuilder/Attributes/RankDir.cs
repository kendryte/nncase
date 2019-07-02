namespace DotBuilder.Attributes
{
    public class RankDir : Attribute, IGraphAttribute
    {
        public RankDir(string value) : base(value)
        {
        }

        public static RankDir TB => new RankDir("TB");
        public static RankDir LR => new RankDir("LR");
        public static RankDir BT => new RankDir("BT");
        public static RankDir RL => new RankDir("RL");
    }
}