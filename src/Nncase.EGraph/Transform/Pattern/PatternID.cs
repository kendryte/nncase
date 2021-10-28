

namespace Nncase.Transform.Pattern
{

    public sealed record ID
    {
        public string id;

        private static int _globalIndex = 0;

        public ID(string Prefix = "Id")
        {

            id = $"{Prefix}_{_globalIndex++}";
        }

        public static implicit operator ID(string Prefix) => new ID(Prefix);

        public static implicit operator ID(int Index) => new ID(Index.ToString());

        public override string ToString()
        {
            return id;
        }
    }

    public partial class Utility
    {
        public static ID GetID(string Prefix = "Id") => new() { };
    }


}