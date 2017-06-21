//
// Created by DY on 17-6-20.
//

#ifndef NLP_CUDA_NATIVELIBRARY_H
#define NLP_CUDA_NATIVELIBRARY_H
#include <string>

namespace NativeLibrary {
    class NativeClass {
    public:
        const std::string& get_property() { return property; }
        void set_property(const std::string& property) { this->property = property; }
        std::string property;
    };
}
#endif //NLP_CUDA_NATIVELIBRARY_H
