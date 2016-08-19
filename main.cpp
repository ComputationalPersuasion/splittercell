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

unsigned int make_cycles(unsigned int num_of_args, unsigned int num_of_cycles, std::default_random_engine gen,
                         std::vector<unsigned int> &cycles) {
    unsigned int in_cycles_so_far;
    do {
        cycles.clear();
        in_cycles_so_far = 0;
        for (unsigned int i = 0; i < num_of_cycles; i++) {
            unsigned int limit = num_of_args * 2 / 3 - in_cycles_so_far;
            if (limit > 1) {
                std::uniform_int_distribution<unsigned int> distribution(2, limit);
                unsigned int num_in_this_cycle = distribution(gen);
                cycles.push_back(num_in_this_cycle);
                in_cycles_so_far += num_in_this_cycle;
            }
        }
    } while (cycles.size() < num_of_cycles);

    return in_cycles_so_far;
}

unsigned int make_connections(unsigned int num_of_args, unsigned int num_of_cycles, unsigned int in_cycles_so_far,
                      unsigned int connectivity, std::default_random_engine gen,
                      std::map<unsigned int, unsigned int> &size_table) {
    std::vector<std::pair<unsigned int, unsigned int>> arcs;
    unsigned int num_solo = num_of_args - in_cycles_so_far;
    std::uniform_int_distribution<unsigned int> attack(0, num_solo + num_of_cycles - 1);
    unsigned int a1, a2;
    bool onetwo, twoone;
    unsigned int trylimit = 1000, current_try = 0;

    for( int i = 0; i < connectivity; i++ ) {
        do {
            a1 = attack(gen);
            a2 = attack(gen);
            onetwo = std::find(arcs.begin(), arcs.end(), std::make_pair(a1, a2)) != arcs.end();
            twoone = std::find(arcs.begin(), arcs.end(), std::make_pair(a2, a1)) != arcs.end();
            current_try++;
        } while ((onetwo || twoone || a1 == a2) && current_try < trylimit);
        arcs.emplace_back(a1, a2);
        if(size_table[a2] == 0)
            size_table[a2] = 1;
        size_table[a2] += 1;
    }

    return num_solo;
}

int main() {
    /********* PARAMETERS ***********/
    unsigned int num_of_args = 50;
    double num_of_connectivity_mean = 100.0, num_of_updates_mean = 100.0;
    std::vector<unsigned int> cycle_sizes = {2, 4, 10};
    std::vector<unsigned int> connectivity {10, 30};
    std::vector<unsigned int> updates {1, 10, 50, 100};
    /********* PARAMETERS ***********/

    std::vector<unsigned int> cycles;
    std::map<unsigned int, unsigned int> size_table;
    unsigned int seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);

    std::cout << num_of_args << " arguments" << std::endl;

    for(unsigned int num_of_cycles : cycle_sizes ) { //Each number of cycles
        std::cout << num_of_cycles << " cycles" << std::endl;

        unsigned int in_cycles_so_far = make_cycles(num_of_args, num_of_cycles, gen, cycles);
        for(unsigned int sizes : cycles)
            std::cout << sizes << " ";
        std::cout << "sizes of cycles" << std::endl;

        for(unsigned num_of_connections : connectivity) { //Each number of arcs
            std::cout << num_of_connections << " arcs" << std::endl;
            std::vector<std::chrono::nanoseconds> conn_mean_time(updates.size(), std::chrono::nanoseconds::zero());
            for(int conn_mean = 0; conn_mean < num_of_connectivity_mean; conn_mean++) {
                size_table.clear();

                for(unsigned int i = 0; i < cycles.size(); i++)
                    size_table[i] = cycles[i];
                unsigned int num_solo = make_connections(num_of_args, num_of_cycles, in_cycles_so_far,
                                                         num_of_connections, gen, size_table);
                unsigned int num_tables = num_solo + num_of_cycles;

                std::uniform_int_distribution<unsigned int> distribution(0, num_tables - 1);
                for(unsigned int index_update = 0; index_update < updates.size(); index_update++) { //Each number of updates
                    std::chrono::nanoseconds time = std::chrono::nanoseconds::zero();
                    for(unsigned int update_mean = 0; update_mean < num_of_updates_mean; update_mean++) {
                        for (unsigned int i = 0; i < updates[index_update]; i++) {
                            unsigned int arg_to_update = distribution(gen);

                            std::vector<unsigned int> vec((unsigned int) pow(2, size_table[arg_to_update]), 0);
                            auto begin = std::chrono::steady_clock::now();
                            update(vec);
                            auto end = std::chrono::steady_clock::now();
                            time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                        }
                    }
                    conn_mean_time[index_update] += time;
                }
            }

            for(unsigned int index_updates = 0; index_updates < updates.size(); index_updates++)
                std::cout << (long)(conn_mean_time[index_updates].count() / (num_of_updates_mean * num_of_connectivity_mean)) << " ns for " << updates[index_updates] << " updates" << std::endl;
        }
    }

    return 0;
}
