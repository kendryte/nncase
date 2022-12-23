// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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
