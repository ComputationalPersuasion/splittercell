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
        distribution(std::vector<std::unique_ptr<flock>> &flocks);
        distribution(const std::vector<unsigned int> &arguments);
        distribution(const distribution &other);
        /* Accessors */
        std::unordered_map<unsigned int, double> operator[](const std::vector<unsigned int> &arguments);
        flock* get_flock(unsigned int f) const {return _flocks[f].get();}
        void set_probabilities(unsigned int f, const std::vector<double> &probabilities) {
            _flocks[f]->set_probabilities(probabilities);
            for(auto arg : _flocks[f]->conditioned())
                _cache_is_valid[arg] = false;
        }
        void disable_mt() {_mt = false;}
        /* Modifiers */
        void refine(unsigned int argument, bool positive, double coefficient) {
            _flocks[_mapping[argument]]->refine(argument, positive, coefficient);
            _cache_is_valid[argument] = false;
        }
        void fast_refine(unsigned int argument, bool positive, double coefficient) {
          if(!_cache_is_valid[argument])
              throw std::invalid_argument("Cannot fast update " + std::to_string(argument) + " cache is invalid.");
          if(positive)
              _belief_cache[argument] += coefficient * (1 - _belief_cache[argument]);
          else
              _belief_cache[argument] *= (1 - coefficient);
        }
        std::unique_ptr<flock> marginalize(unsigned int f, const std::vector<unsigned int> &args_to_keep) {
            return _flocks[f]->marginalize(args_to_keep);
        }

        std::string to_str() const;

    private:
        std::vector<std::unique_ptr<flock>> _flocks;
        std::unordered_map<unsigned int, unsigned int> _mapping;
        std::unordered_map<unsigned int, double> _belief_cache;
        std::unordered_map<unsigned int, bool> _cache_is_valid;
        bool _mt;

        void find_conditioning(unsigned int argument, std::set<unsigned int> &conditioning) const;
        std::unique_ptr<flock> find_and_combine(const std::vector<unsigned int> &arguments) const;
    };
}

#endif //SPLITTERCELL_DISTRIBUTION_H
