// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase
{
    public record HuggingFaceOptions
    {
        public bool OutputAttentions { get; set; }

        public bool OutputHiddenStates { get; set; }

        public bool UseCache { get; set; }

        public static HuggingFaceOptions Default => new();
    }
}
