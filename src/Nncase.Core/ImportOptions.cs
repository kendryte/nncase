// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;

namespace Nncase
{
    /// <summary>
    /// Import options.
    /// </summary>
    public sealed record ImportOptions
    {
        /// <summary>
        /// Gets or sets huggingface options.
        /// </summary>
        public HuggingFaceOptions HuggingFaceOptions { get; set; } = HuggingFaceOptions.Default;
    }
}
