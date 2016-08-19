#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <map>
#include <utility>
#include <algorithm>

void update(std::vector<unsigned int> &v) {
    for( unsigned long i = 0; i < v.size(); i++ )
        v[i] += 1;
}

int main() {
    /*   PARAMETERS */
    std::vector<unsigned int> var_args {25, 50, 75, 100};
    unsigned int min_cycle_size = 1;
    unsigned int max_cycle_size = 15;
    unsigned int num_updates = 20;
    double connectivity_factor = 0.5;
    unsigned int instance_mean = 1000;
    unsigned int update_mean = 1000;
    /*   PARAMETERS */

    unsigned int seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<unsigned int> size_distrib(min_cycle_size, max_cycle_size);

    for(auto num_cycles : var_args) {
        auto time = std::chrono::nanoseconds::zero();

        for(unsigned int instance_num = 0; instance_num < instance_mean; instance_num++) {
            std::vector<unsigned int> sizes;
            unsigned int args_left = num_cycles;
            while (args_left > 0) {
                unsigned int size;
                do {
                    size = size_distrib(gen);
                } while (size > args_left);
                sizes.push_back(size);
                args_left -= size;
            }

            std::uniform_int_distribution<unsigned int> connection_distrib(0, sizes.size() - 1);
            for (unsigned int i = 0; i < (unsigned int) (num_cycles * connectivity_factor); i++)
                sizes[connection_distrib(gen)]++;

            std::vector<std::vector<unsigned int>> proba_vectors(sizes.size());
            for (unsigned int i = 0; i < sizes.size(); i++)
                proba_vectors[i] = std::vector<unsigned int>(sizes[i], 0);

            std::discrete_distribution<unsigned int> update_distrib(sizes.begin(), sizes.end());

            for(unsigned int i = 0; i < update_mean; i++) {
                for (unsigned int i = 0; i < num_updates; i++) {
                    unsigned int aim = update_distrib(gen);
                    auto begin = std::chrono::steady_clock::now();
                    update(proba_vectors[aim]);
                    auto end = std::chrono::steady_clock::now();
                    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                }
            }
        }
        std::cout << num_cycles << ";" << (long)(time.count() / (instance_mean * update_mean)) << std::endl; //nanoseconds
    }
}