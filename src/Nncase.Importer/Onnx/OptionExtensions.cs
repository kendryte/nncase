// using NetFabric.Hyperlinq;
using LanguageExt;

namespace Nncase.Importer
{
    public static class OptionExtensions
    {
        public static T Or<T>(this Option<T> v, T defaultValue)
        {
            return v.Match(x => x, () => defaultValue);
        }
    }
}