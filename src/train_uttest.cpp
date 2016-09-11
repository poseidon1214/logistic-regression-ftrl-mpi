#include "../include/owlqn.h"
#include <vector>
#include <gtest/gtest.h>

class TrainTest: public testing::Test{
    public: 
        //TrainTest() {}
        //virtual ~TrainTest(){}
        LR* lr;
        virtual void SetUp(){
            lr = new LR();
        }
        void TearDown(){
            delete lr;
        }
};
/*
TEST_F(TrainTest, test_test)
{
    float a;
    a = 1.0;
    ASSERT_EQ(1.0, lr->sigmoid(a));
}
*/
TEST_F(TrainTest, test_testb){
    float *w = new float[3];
    w[0] = 1.0;
    w[1] = 1.0;
    w[2] = 1.0;
    float* g = new float[3];
    g[0] = 1.0;
    g[1] = 1.0;
    g[2] = 1.0;

    lr->loss_function_gradient(w, g);  
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}
