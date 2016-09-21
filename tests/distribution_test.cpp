#include "gtest/gtest.h"
#include "distribution.h"

class DistributionTest: public ::testing::Test {
public:
    virtual void SetUp() {
        dist = new splittercell::Distribution({splittercell::Flock({0,1}, {0})});
        dist->set_flock_probabilities(0, {0.1, 0.2, 0.1, 0.6});
    }

    virtual void TearDown() {
        delete dist;
    }

    splittercell::Distribution *dist;
};

TEST_F(DistributionTest, Refine1A) {
    EXPECT_STREQ(dist->to_str().c_str(), "0.1 0.2 0.1 0.6 ");
    dist->refine(0, true, 1.0);
    EXPECT_STREQ(dist->to_str().c_str(), "0 0.3 0 0.7 ");
}

TEST_F(DistributionTest, Refine1NotA) {
    EXPECT_STREQ(dist->to_str().c_str(), "0.1 0.2 0.1 0.6 ");
    dist->refine(0, false, 1.0);
    EXPECT_STREQ(dist->to_str().c_str(), "0.3 0 0.7 0 ");
}

TEST_F(DistributionTest, Refine75A) {
    EXPECT_STREQ(dist->to_str().c_str(), "0.1 0.2 0.1 0.6 ");
    dist->refine(0, true, 0.75);
    EXPECT_STREQ(dist->to_str().c_str(), "0.025 0.275 0.025 0.675 ");
}

TEST_F(DistributionTest, Refine1B) {
    EXPECT_STREQ(dist->to_str().c_str(), "0.1 0.2 0.1 0.6 ");
    dist->refine(1, true, 1.0);
    EXPECT_STREQ(dist->to_str().c_str(), "0 0 0.2 0.8 ");
}