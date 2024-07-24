//
// Created by Trung Luu on 7/20/24.
//

#pragma once

namespace Toygrad::Tensor {
    using real = float;

    struct Range {
        size_t beg;
        size_t end;
        size_t step;
    };
}
