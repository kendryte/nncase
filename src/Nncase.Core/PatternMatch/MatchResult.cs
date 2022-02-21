using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.PatternMatch;

/// <summary>
/// Match result.
/// </summary>
public class MatchResult : IMatchResult
{
    private readonly Dictionary<IPattern, object> _matches = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="MatchResult"/> class.
    /// </summary>
    /// <param name="matches">Matches.</param>
    public MatchResult(Dictionary<IPattern, object> matches)
    {
        _matches = matches;
    }

    /// <inheritdoc/>
    public object this[IPattern pattern] => _matches[pattern];
}
