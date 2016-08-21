#include <limits>
#include <stdexcept>
#include "distribution.h"

using namespace splittercell;

template <typename T>
Distribution<T>::Distribution(const Graph &graph, const std::vector<std::vector<std::string>> &flocks) {
    unsigned int flock_index = 0;
    for(auto flock : flocks) {
        if(flock.size() > std::numeric_limits<unsigned long>::digits)
            throw std::overflow_error("Too many arguments in the flock.");
        unsigned int order_index = 0;
        for(auto arg : flock) {
            _mapping.emplace(arg, flock_index, order_index);
            order_index++;
        }
        flock_index++;
    }
}

template <typename T>
Distribution<T>::Distribution(const Distribution<T> &other) :
        _distribution(other._distribution),
        _mapping(other._mapping) {}
