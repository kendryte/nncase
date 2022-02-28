using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.PatternMatch;

/// <summary>
/// Match result.
/// </summary>
public class MatchResult : IMatchResult
{
    private readonly IReadOnlyDictionary<IPattern, object> _patternMap;

    private readonly Dictionary<string, object> _stringMap;

    /// <summary>
    /// Initializes a new instance of the <see cref="MatchResult"/> class.
    /// </summary>
    /// <param name="matches">Matches.</param>
    public MatchResult(Dictionary<IPattern, object> matches)
    {
        _patternMap = matches;
        _stringMap = (from kv in matches
                      where kv.Key.Name is not null
                      select (kv.Key.Name, kv.Value)).ToDictionary(t => t.Name, t => t.Value);
    }

    /// <inheritdoc/>
    public object this[IPattern pattern] => _patternMap[pattern];


    /// <inheritdoc/>
    public object this[string name] => _stringMap[name];
}
