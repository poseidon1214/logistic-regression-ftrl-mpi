#!/bin/bash
process_number=3
Ip=("10.101.2.89" "10.101.2.90")
for ip in ${Ip[@]}
do
    ssh worker@$ip rm /home/worker/xiaoshu/logistic-regression-ftrl-mpi/train
done
scp train worker@10.101.2.89:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/.
scp train worker@10.101.2.90:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/.
#mpirun -f ./hosts -np $process_number ./train ftrl 30 5 ./data/v2v_train ./data/v2v_test
mpirun -f ./hosts -np $process_number ./train ftrl 11 1000 1.0 1.0 1.0 1.0 1.0 ./data/traindataold ./data/testdataold
#mpirun -f ../hosts -np $process_number ./train ftrl ./data/agaricus.txt.train ./data/agaricus.txt.test
