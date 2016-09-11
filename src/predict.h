#ifndef PREDICT_H_
#define PREDICT_H_

#include <iostream>
#include <fstream>
#include <string.h>
#include "predict.h"

#define MPI_NON_CLK_TAG 99
#define MPI_CLK_TAG 199

typedef struct{
    float clk;
    float nclk;
    long idx;
} clkinfo;

class Predict{
    public:
    Predict(Load_Data* load_data, int total_num_proc, int my_rank) 
            : data(load_data), nproc(total_num_proc), rank(my_rank){
        pctr = 0.0;
        MAX_ARRAY_SIZE = 1000;
        MAX_BUF_SIZE = 2048;
        g_all_non_clk = new float[MAX_ARRAY_SIZE];
        g_all_clk = new float[MAX_ARRAY_SIZE];
        g_nclk = new float[MAX_ARRAY_SIZE];
        g_clk = new float[MAX_ARRAY_SIZE];
    }
    ~Predict(){}

    void predict(std::vector<float> glo_w){
        std::cout<<"glo_w size "<<glo_w.size()<<std::endl;
        std::vector<float> predict_result;
        for(int i = 0; i < data->fea_matrix.size(); i++) {
	        float x = 0.0;
            for(int j = 0; j < data->fea_matrix[i].size(); j++) {
                int idx = data->fea_matrix[i][j].idx;
                float val = data->fea_matrix[i][j].val;
                x += glo_w[idx] * val;
            }
        
            if(x < -30){
                pctr = 1e-6;
            }
            else if(x > 30){
                pctr = 1.0;
            }
            else{
                double ex = pow(2.718281828, x);
                pctr = ex / (1.0 + ex);
            }

            int id = int(pctr*MAX_ARRAY_SIZE);
            clkinfo clickinfo;
            clickinfo.clk = data->label[i];
            clickinfo.nclk = 1 - data->label[i];
            clickinfo.idx = id;
            result_list.push_back(clickinfo);
        }
    
        for(size_t j = 0; j < predict_result.size(); j++){
	        std::cout<<predict_result[j]<<"\t"<<1 - data->label[j]<<"\t"<<data->label[j]<<std::endl;
        }
    }

    int merge_clk(){//merge local node`s clk
        memset(g_nclk, 0, MAX_ARRAY_SIZE * sizeof(float));
        memset(g_clk, 0, MAX_ARRAY_SIZE * sizeof(float));
        int cnt = result_list.size();
        for(int i = 0; i < cnt; i++){
            int idx = result_list[i].idx;
            g_nclk[idx] += result_list[i].nclk;
            g_clk[idx] += result_list[i].clk;
        }
        return 0;
    }

    int auc_cal(float* all_clk, float* all_nclk, double& auc_res){
            double clk_sum = 0.0;
            double nclk_sum = 0.0;
            double old_clk_sum = 0.0;
            double clksum_multi_nclksum = 0.0;
            double auc = 0.0;
            auc_res = 0.0;
            for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                    old_clk_sum = clk_sum;
                    clk_sum += all_clk[i];
                    nclk_sum += all_nclk[i];
                    auc += (old_clk_sum + clk_sum) * all_nclk[i] / 2;
            }
            clksum_multi_nclksum = clk_sum * nclk_sum;
            auc_res = auc/(clksum_multi_nclksum);
    }

    int mpi_auc(int nprocs, int rank, double& auc){
        MPI_Status status;
        if(rank != MASTER_ID){
            MPI_Send(g_nclk, MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, MPI_NON_CLK_TAG, MPI_COMM_WORLD);
            MPI_Send(g_clk, MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, MPI_CLK_TAG, MPI_COMM_WORLD);
        }
        else if(rank == MASTER_ID){
            for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                g_all_non_clk[i] = g_nclk[i];
                g_all_clk[i] = g_clk[i];
            }
            for(int i = 1; i < nprocs; i++){
                MPI_Recv(g_nclk, MAX_ARRAY_SIZE, MPI_FLOAT, i, MPI_NON_CLK_TAG, MPI_COMM_WORLD, &status);
                MPI_Recv(g_clk, MAX_ARRAY_SIZE, MPI_FLOAT, i, MPI_CLK_TAG, MPI_COMM_WORLD, &status);
                for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                    g_all_non_clk[i] += g_nclk[i];
                    g_all_clk[i] += g_clk[i];
                }
            }
            auc_cal(g_all_non_clk, g_all_clk, auc);
        }
    }

    void run(std::vector<float> w){
        predict(w);
        double total_clk = 0.0;
        double total_nclk = 0.0;
        double auc = 0.0;
        double total_auc = 0.0;

        merge_clk();
        mpi_auc(nproc, rank, auc);

        if(MASTER_ID == rank){
                printf("AUC = %lf\n", auc);
        }

    }

    private:
    Load_Data* data;
    std::vector<clkinfo> result_list;
    int MAX_ARRAY_SIZE;
    int MAX_BUF_SIZE;
    float* g_all_non_clk;
    float* g_all_clk;
    float* g_nclk;
    float* g_clk;
    double g_total_clk;
    double g_total_nclk;
    float pctr;
    //MPI process info
    int nproc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world
};
#endif
