#ifndef SPLITTERCELL_DISTRIBUTION_H
#define SPLITTERCELL_DISTRIBUTION_H

#include <vector>
#include <map>
#include <utility>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <sstream>

typedef std::pair<std::vector<unsigned int>, std::vector<unsigned int>> Flock;

namespace splittercell {
    class Distribution {
    public:
        Distribution(const std::vector<Flock> &flocks);

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

    Distribution::Distribution(const std::vector<Flock> &flocks) :
            _distribution(flocks.size()) {
        unsigned int flock_index = 0;
        for (auto flock : flocks) {
            unsigned int arg_index = 0, flock_size = (unsigned int) (flock.first.size() + flock.second.size());
            if (flock_size > std::numeric_limits<unsigned int>::digits)
                throw std::overflow_error("Too many arguments in the flock.");
            _sizes.emplace(flock_index, flock_size);
            for (auto conditioned : flock.first) {
                if (_mapping.find(conditioned) != _mapping.end())
                    throw std::invalid_argument("An argument cannot be in different flocks.");
                _mapping.emplace(conditioned, std::make_pair(flock_index, arg_index));
                arg_index++;
            }
            _distribution[flock_index] = std::vector<double>((unsigned int)(1 << flock_size), 1.0 / (1 << flock_size));
            flock_index++;
        }
    }

    void Distribution::refine(unsigned int argument, bool positive, double coefficient) {
        unsigned int flock_ind, arg_ind_in_flock;
        std::tie(flock_ind, arg_ind_in_flock) = _mapping[argument];
        std::vector<double> &flock = _distribution[flock_ind];
        unsigned int num_of_args = _sizes[flock_ind];

        for (unsigned int i = 0; i < flock.size(); i++) {
            boost::dynamic_bitset<> binary(num_of_args, i);
            if (binary.test(arg_ind_in_flock) == positive) {
                unsigned int opposite = (unsigned int) binary.flip(arg_ind_in_flock).to_ulong();
                double opposite_val = flock[opposite];
                flock[opposite] *= (1 - coefficient);
                flock[i] += coefficient * opposite_val;
            }
        }
    }

    std::string Distribution::to_str() const {
        std::stringstream ss;
        for (auto distribution : _distribution)
            for (double val : distribution) ss << val << " ";
        return ss.str();
    }

    double Distribution::operator[](const unsigned int &argument) const {

    }

    std::ostream &operator<<(std::ostream &os, const Distribution &dist) {
        os << dist.to_str() << std::endl;
        return os;
    }
}


#endif //SPLITTERCELL_DISTRIBUTION_H
