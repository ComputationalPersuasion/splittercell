#ifndef SPLITTERCELL_DISTRIBUTION_H
#define SPLITTERCELL_DISTRIBUTION_H

#include <vector>
#include <map>
#include <utility>
#include <iostream>

typedef std::pair<std::vector<unsigned int>, std::vector<unsigned int>> Flock;

namespace splittercell {
    class Distribution {
    public:
        Distribution(const std::vector<Flock> &flocks);
        Distribution(const Distribution &other);

        void refine(unsigned int argument, bool positive, double coefficient);

        void set_flock_probabilities(unsigned int flock, const std::vector<double> &probabilities) { _distribution[flock] = probabilities; }
        const std::vector<std::vector<double>>& distribution() const { return _distribution; }

        std::string to_str() const;

        double operator[](const std::string &model) const;
        double operator[](const unsigned int &argument) const;

    private:
        std::vector<std::vector<double>> _distribution;
        std::map<unsigned int, std::pair<unsigned int, unsigned int>> _mapping;
        std::map<unsigned int, unsigned int> _sizes;
    };

    std::ostream& operator<<(std::ostream &os, const Distribution &dist);
}


#endif //SPLITTERCELL_DISTRIBUTION_H
