using System.Collections.Generic;
using System.Linq;
using Nncase.IR;

namespace Nncase.Passes.BufferSchedule;

public static class BufferScheduleExtensions
{

    public static IEnumerable<Expr> GetUsers(this Call call)
    {
        var hs = new HashSet<Expr>(ReferenceEqualityComparer.Instance);
        hs.UnionWith(call.Users.Where(e => e is not BaseFunction).ToArray().Select(e => e switch { IR.Tuple tp => tp.Fields.ToArray(), _ => new[] { e } }).SelectMany(i => i));
        return hs;
    }

}