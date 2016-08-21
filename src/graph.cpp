#include <memory>
#include "graph.h"

using namespace splittercell;

void Graph::attacks(const std::string &a1, const std::string &a2) {
    _attacks.insert(std::make_pair(a1, a2));
    _is_atked_by.insert(std::make_pair(a2, a1));
    _arguments.insert({a1, a2});
}

