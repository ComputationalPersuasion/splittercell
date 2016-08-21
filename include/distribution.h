#ifndef SPLITTERCELL_DISTRIBUTION_H
#define SPLITTERCELL_DISTRIBUTION_H

#include <vector>
#include <map>
#include <string>
#include <memory>
#include "graph.h"

namespace splittercell {
    template <typename T>
    class Distribution {
    public:
        Distribution(const Graph &graph, const std::vector<std::vector<std::string>> &flocks);
        Distribution(const Distribution<T> &other);

        const T &operator[](const std::string &model) const;
        T &operator[](const std::string &model);

    private:
        std::vector<std::vector<T>> _distribution;
        std::multimap<std::string, std::pair<unsigned long, unsigned long>> _mapping;
    };
}


#endif //SPLITTERCELL_DISTRIBUTION_H
