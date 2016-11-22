#include <stdexcept>
#include <limits>
#include <sstream>
#include "distribution.h"

namespace splittercell {
    distribution::distribution(std::vector<std::unique_ptr<flock>> &flocks) : _flocks(std::move(flocks)), _mt(true) {
        unsigned int flock_index = 0;
        for (auto &f : _flocks) {
            for (auto conditioned : f->conditioned()) {
                if (_mapping.find(conditioned) != _mapping.end())
                    throw std::invalid_argument("An argument cannot be in different flocks.");
                _mapping[conditioned] = flock_index;
                if(f->uniform()) {
                    _belief_cache[conditioned] = 0.5;
                    _cache_is_valid[conditioned] = true;
                } else
                    _cache_is_valid[conditioned] = false;
            }
            flock_index++;
        }
    }

    distribution::distribution(const distribution &other) : _mapping(other._mapping), _belief_cache(other._belief_cache),
                                                            _cache_is_valid(other._cache_is_valid) {
        for(auto &f : other._flocks)
            _flocks.push_back(std::make_unique<flock>(*f));
    }

    std::unordered_map<unsigned int, double> distribution::operator[](const std::vector<unsigned int> &arguments) const {
        std::unordered_map<unsigned int, double> beliefs;
        std::unique_ptr<flock> f = nullptr;
        for(auto arg : arguments) {
            if(_cache_is_valid.at(arg))
                beliefs[arg] = _belief_cache.at(arg);
            else {
                if(f == nullptr)
                    f = find_and_combine(arguments);
                beliefs[arg] = f->marginalize({arg})->distribution()[1];
            }
        }

        return beliefs;
    }

    void distribution::find_conditioning(unsigned int argument, std::set<unsigned int> &conditioning) const {
        auto f = _flocks[_mapping.at(argument)].get();
        for(auto cond : f->conditioning()) {
            conditioning.insert(cond);
            find_conditioning(cond, conditioning);
        }
    }

    std::unique_ptr<flock> distribution::find_and_combine(const std::vector<unsigned int> &arguments) const {
        std::set<unsigned int> conditioning_args;
        std::set<flock*> conditioning_flocks;
        for(auto arg : arguments) {
            find_conditioning(arg, conditioning_args);
            conditioning_flocks.insert(_flocks[_mapping.at(arg)].get());
        }
        for(auto arg : conditioning_args)
            conditioning_flocks.insert(_flocks[_mapping.at(arg)].get());

        auto it = conditioning_flocks.cbegin();
        auto combined = *it;
        std::unique_ptr<flock> ptrcombined = nullptr, ptrnext = nullptr;
        ++it;
        unsigned int limit = std::numeric_limits<unsigned int>::digits - 2;
        while(it != conditioning_flocks.cend()) {
            auto next = *it;
            if(combined->size() + next->size() > limit) {
                auto all = std::vector<unsigned int>(arguments);
                all.insert(all.cend(), conditioning_args.cbegin(), conditioning_args.cend());
                ptrcombined = combined->marginalize(all);
                combined = ptrcombined.get();
                ptrnext = next->marginalize(all);
                next = ptrnext.get();
            }
            ptrcombined = combined->combine(next, _mt);
            combined = ptrcombined.get();
            ++it;
        }

        return combined->marginalize(arguments);
    }

    std::string distribution::to_str() const {
        std::stringstream ss;
        for (auto &f : _flocks)
            ss << f->to_str() << " ";
        std::string s = ss.str();
        s.pop_back();
        return s;
    }
}
