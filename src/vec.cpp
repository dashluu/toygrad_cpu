#include "vec.h"

namespace Toygrad::Tensor {
    std::ostream &operator<<(std::ostream &stream, const Vec &vec) {
        for (size_t i = 0; i < vec.size; i++) {
            stream << vec.buff[i];

            if (i < vec.size - 1) {
                stream << " ";
            }
        }

        return stream;
    }
}
