#ifndef SPLITTERCELL_GRAPH_H
#define SPLITTERCELL_GRAPH_H

#include <map>
#include <unordered_set>
#include <string>
#include <memory>

namespace splittercell {
    class Graph {
    public:
        void attacks(const std::string &a1, const std::string &a2);
        unsigned long get_num_of_args() const {return _arguments.size();}

    private:
        std::multimap<std::string, std::string> _attacks, _is_atked_by;
        std::unordered_set<std::string> _arguments;
    };
}


#endif //SPLITTERCELL_GRAPH_H
