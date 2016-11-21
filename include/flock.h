#ifndef SPLITTERCELL_FLOCK_H
#define SPLITTERCELL_FLOCK_H

#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <unordered_map>
#include <utility>

namespace splittercell {
    class flock {
    public:
        /* Constructors */
        flock(const std::vector<unsigned int> &args, const std::vector<unsigned int> &cond = {}, const std::vector<double> &distribution = {});
        flock(const flock &other);
        /* Accessors */
        unsigned int size() const {return _size;}
        const std::vector<double> &distribution() const {return _distribution;}
        void set_probabilities(const std::vector<double> &probabilities) {_distribution = probabilities; _uniform = false;}
        const std::vector<unsigned int> &conditioned() const {return _conditioned;}
        const std::vector<unsigned int> &conditioning() const {return _conditioning;}
        bool uniform() const {return _uniform;}
        /* Modifiers*/
        void refine(unsigned int argument, bool positive, double coefficient, bool mt = true);
        std::unique_ptr<flock> marginalize(const std::vector<unsigned int> &args_to_keep, bool mt = true) const;
        void marginalize_self(const std::vector<unsigned int> &args_to_keep, bool mt = true);
        std::unique_ptr<flock> combine(const flock * const f, bool mt = true) const;

        std::string to_str() const;
        bool operator==(const flock &other) const {return (_conditioned == other._conditioned) &&
                    (_conditioning == other._conditioning) && (_distribution == other._distribution); }

    private:
        std::vector<unsigned int> _conditioned, _conditioning;
        std::vector<double> _distribution;
        std::unordered_map<unsigned int, unsigned int> _mapping;
        unsigned int _size;
        bool _uniform;

        void map_arguments();
        std::vector<double> marginalized_distribution(const std::vector<unsigned int> &args_to_keep, bool mt) const;
        void mt_refine(unsigned int index, bool positive, double coefficient, unsigned int startindex, unsigned int endindex);
        void mt_marginalize(std::vector<double> &distribution, const std::map<unsigned int, unsigned int> &mapping, unsigned int startindex, unsigned int endindex) const;
        void mt_combine(flock * const combinedflock, const flock * const f, const std::unordered_map<unsigned int,
                std::pair<unsigned int, unsigned int>> &splitindex, unsigned int startindex, unsigned int endindex) const;
        void perform_mt(unsigned int bound, std::function<void(unsigned int, unsigned int)> t) const;
    };
}

#endif //SPLITTERCELL_FLOCK_H
