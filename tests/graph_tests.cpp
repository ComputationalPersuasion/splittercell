#include "gtest/gtest.h"
#include "graph.h"

TEST(Graph, Attacks) {
    splittercell::Graph g;
    g.attacks("a", "b");
}
