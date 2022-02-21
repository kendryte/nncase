using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Transform;

internal class RewriteProvider : IRewriteProvider
{
    public Expr Rewrite(Expr expr, IEnumerable<IRewriteRule> rules, RunPassOptions options)
    {
        var rewriter = new DataflowRewriter();
        return rewriter.Rewrite(expr, rules, options);
    }
}
