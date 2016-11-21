#include <boost/dynamic_bitset.hpp>
#include <algorithm>
#include <stdexcept>
#include <future>
#include <sstream>
#include "flock.h"

inline void set_bin_value(unsigned int &end, unsigned int i, unsigned int mask, unsigned int where) {
      end |= ((bool)(i & (1 << mask)) << where);
}

namespace splittercell {
    flock::flock(const std::vector<unsigned int> &args, const std::vector<unsigned int> &cond, const std::vector<double> &distribution) :
            _conditioned(args), _conditioning(cond), _distribution(distribution), _size(args.size() + cond.size()), _uniform(false) {
        unsigned int limit = std::numeric_limits<unsigned int>::digits;
        if(_size > (limit - 2))
            throw std::overflow_error("Too many arguments in the flock.");
        unsigned int num_of_models = (unsigned int)(1 << _size);
        double initial_belief      = 1.0 * (1 << _conditioning.size()) / num_of_models;
        if(_distribution.empty()) {
            _distribution = std::vector<double>(num_of_models, initial_belief);
            _uniform      = true; //If the distribution is not uniform, we cannot cache 0.5 as a belief for the arguments in this flock
        }

        map_arguments();
    }

    flock::flock(const flock &other) : _conditioned(other._conditioned), _conditioning(other._conditioning), _distribution(other._distribution),
                                       _mapping(other._mapping), _size(other._size), _uniform(other._uniform) {}

    std::string flock::to_str() const {
        std::stringstream ss;
        for (double val : _distribution) ss << val << " ";
        std::string s = ss.str();
        s.pop_back();
        return s;
    }

    void flock::refine(unsigned int argument, bool positive, double coefficient, bool mt) {
        unsigned int index = _mapping[argument];
        if(index >= _conditioned.size())
            throw std::invalid_argument("Only conditioned arguments can be refined.");
        if(_size < 15 || !mt)
            mt_refine(index, positive, coefficient, 0, _distribution.size());
        else //Beware of the potential race condition
            perform_mt(_distribution.size(), std::bind(&flock::mt_refine, this, index, positive, coefficient, std::placeholders::_1, std::placeholders::_2));
    }

    void flock::mt_refine(unsigned int index, bool positive, double coefficient, unsigned int startindex, unsigned int endindex) {
        for (unsigned int i = startindex; i < endindex; i++) {
            if((bool)(i & (1 << index)) == positive) { //If the model satisfies argument+side of update
                unsigned int opposite = i ^ (1 << index); //Closest model not satisfying
                double opposite_val = _distribution[opposite];
                _distribution[opposite] *= (1 - coefficient);
                _distribution[i] += coefficient * opposite_val;
            }
        }
    }

    std::vector<double> flock::marginalized_distribution(const std::vector<unsigned int> &args_to_keep, bool mt) const {
        /* New mapping creation (because marginalization put holes in the previous one) */
        unsigned int index = 0;
        std::map<unsigned int, unsigned int> mapping;
        if(args_to_keep == _conditioned)
            return _distribution;
        for(unsigned int arg : args_to_keep)
            if(_mapping.find(arg) != _mapping.end())
                mapping[_mapping.at(arg)] = index++;
        for(unsigned int arg : _conditioning)
            if(_mapping.find(arg) != _mapping.end() && mapping.find(_mapping.at(arg)) == mapping.cend())
                mapping[_mapping.at(arg)] = index++;

        /* Actual marginalization */
        unsigned int marginalized_size = (unsigned int)(1 << mapping.size());
        auto distribution = std::vector<double>(marginalized_size, 0.0);

        if(_size < 15 || !mt)
            mt_marginalize(distribution, mapping, 0, _distribution.size());
        else //Beware of the race condition
            perform_mt(_distribution.size(), std::bind(&flock::mt_marginalize, this, std::ref(distribution), std::cref(mapping), std::placeholders::_1, std::placeholders::_2));

        return distribution;
    }

    void flock::mt_marginalize(std::vector<double> &distribution, const std::map<unsigned int, unsigned int> &mapping, unsigned int startindex, unsigned int endindex) const {
        for(unsigned int i = startindex; i < endindex; i++) {
            unsigned int end = 0;
            for(auto map : mapping)
                set_bin_value(end, i, map.first, map.second);
            distribution[end] += _distribution[i];
        }
    }

    std::unique_ptr<flock> flock::marginalize(const std::vector<unsigned int> &args_to_keep, bool mt) const {
        return std::make_unique<flock>(args_to_keep, _conditioning, marginalized_distribution(args_to_keep, mt));
    }

    void flock::marginalize_self(const std::vector<unsigned int> &args_to_keep, bool mt) {
        _conditioned  = args_to_keep;
        _distribution = marginalized_distribution(args_to_keep, mt);
        map_arguments();
    }

    std::unique_ptr<flock> flock::combine(const flock * const f, bool mt) const {
        /* Combined flock creation */
        std::vector<unsigned int> conditioned, conditioning;
        conditioned.insert(conditioned.end(), _conditioned.cbegin(), _conditioned.cend());
        conditioned.insert(conditioned.end(), f->_conditioned.cbegin(), f->_conditioned.cend());
        /* Remove the conditioning arguments that are conditioned in the other flock (Bayes' rule) */
        std::copy_if(_conditioning.cbegin(), _conditioning.cend(), std::back_inserter(conditioning),
                     [f](unsigned int i){return std::find(f->_conditioned.cbegin(), f->_conditioned.cend(), i) == f->_conditioned.cend();});
        std::copy_if(f->_conditioning.cbegin(), f->_conditioning.cend(), std::back_inserter(conditioning),
                     [this](unsigned int i){return std::find(this->_conditioned.cbegin(), this->_conditioned.cend(), i) == this->_conditioned.cend();});
        auto combinedflock = std::make_unique<flock>(conditioned, conditioning);
        unsigned int limit = std::numeric_limits<unsigned int>::digits - 2;
        if(combinedflock->size() > limit)
            throw std::overflow_error("Too many arguments in the final combined flock.");

        /* Mapping between combined flock indexes and split flock index */
        conditioned.insert(conditioned.end(), conditioning.cbegin(), conditioning.cend()); //Just to have the whole set of arguments
        std::vector<std::pair<unsigned int, unsigned int>> splitindex;
        splitindex.resize(conditioned.size());
        for(auto arg : conditioned) {
            auto itindexself = _mapping.find(arg);
            unsigned int indexself = (itindexself == _mapping.end()) ? 0 : itindexself->second + 1; //Careful, mandatory to keep unsigned
            auto itindexother = f->_mapping.find(arg);
            unsigned int indexother = (itindexother == f->_mapping.end()) ? 0 : itindexother->second + 1; //Careful
            splitindex[combinedflock->_mapping[arg]] = std::make_pair<>(indexself, indexother);
        }

        auto combinedptr = combinedflock.get();
        if(combinedflock->size() < 15 || !mt)
            mt_combine(combinedptr, f, splitindex, 0, 1 << combinedflock->size());
        else
            perform_mt(combinedflock->distribution().size(), std::bind(&flock::mt_combine, this, std::cref(combinedptr),
                                                             std::cref(f), std::cref(splitindex), std::placeholders::_1, std::placeholders::_2));

        return combinedflock;
    }

    void flock::mt_combine(flock * const combinedflock, const flock * const f, const std::vector<std::pair<unsigned int, unsigned int>> &splitindex, unsigned int startindex, unsigned int endindex) const {
        for(unsigned int i = startindex; i < endindex; i++) {
            unsigned int end1 = 0, end2 = 0;
            for(unsigned int j = 0; j < combinedflock->size(); j++) {
                auto indexes = splitindex[j];
                if(indexes.first != 0)
                    set_bin_value(end1, i, j, indexes.first-1); //-1 because +1 above
                if(indexes.second != 0)
                    set_bin_value(end2, i, j, indexes.second-1); //-1 because +1 above
            }
            combinedflock->_distribution[i] = _distribution[end1] * f->_distribution[end2];
        }
    }

    void flock::perform_mt(unsigned int bound, std::function<void(unsigned int, unsigned int)> t) const {
        unsigned int numThreads = std::thread::hardware_concurrency();
        unsigned int range = bound / numThreads;
        std::vector<std::future<void>> futures;
        for(unsigned int i = 0; i < numThreads; i++)
            futures.push_back(std::async(std::launch::async, t, i * range, (i + 1) * range));
        for(auto &f : futures)
          f.get();
    }

    /* Mapping argument <-> index to be (somewhat) order agnostic, except conditioned first, then conditioning */
    void flock::map_arguments() {
        unsigned int index = 0;
        for(unsigned int arg : _conditioned)
            _mapping[arg] = index++;
        for(unsigned int arg : _conditioning)
            _mapping[arg] = index++;
    }
}
