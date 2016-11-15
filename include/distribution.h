#ifndef SPLITTERCELL_DISTRIBUTION_H
#define SPLITTERCELL_DISTRIBUTION_H

#include <vector>
#include <memory>
#include <unordered_map>
#include <set>
#include "flock.h"

namespace splittercell {
    class distribution {
    public:
        /* Constructors */
        distribution(const std::vector<std::shared_ptr<flock>> &flocks);
        distribution(const distribution &other);
        /* Accessors */
        std::unordered_map<unsigned int, double> operator[](const std::vector<unsigned int> &arguments) const;
        std::shared_ptr<flock> get_flock(unsigned int f) const {return _flocks[f];}
        void set_probabilities(unsigned int f, const std::vector<double> &probabilities) {
            _flocks[f]->set_probabilities(probabilities);
            for(auto arg : _flocks[f]->conditioned())
                _cache_is_valid[arg] = false;
        }
        void disable_mt() {_mt = false;}
        /* Modifiers */
        void refine(unsigned int argument, bool positive, double coefficient) {
            _flocks[_mapping[argument]]->refine(argument, positive, coefficient, _mt);
            _cache_is_valid[argument] = false;
        }
        std::shared_ptr<flock> marginalize(unsigned int f, const std::vector<unsigned int> &args_to_keep) {
            return _flocks[f]->marginalize(args_to_keep, _mt);
        }

        std::string to_str() const;

    private:
        std::vector<std::shared_ptr<flock>> _flocks;
        std::unordered_map<unsigned int, unsigned int> _mapping;
        std::unordered_map<unsigned int, double> _belief_cache;
        std::unordered_map<unsigned int, bool> _cache_is_valid;
        bool _mt;

        void find_conditioning(unsigned int argument, std::set<unsigned int> &conditioning) const;
        std::shared_ptr<flock> find_and_combine(const std::vector<unsigned int> &arguments) const;
    };
}

#endif //SPLITTERCELL_DISTRIBUTION_H
