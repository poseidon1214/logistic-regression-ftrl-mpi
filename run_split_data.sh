rm testdataold-0000*
python split_data.py test_data_old.txt 3 testdataold
#python split_data.py test_data.txt 3 test_new
scp testdataold-0000* slave1:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/data
scp testdataold-0000* slave2:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/data

rm traindataold-0000*
python split_data.py train_data_old.txt 3 traindataold
#python split_data.py test_data.txt 3 test_new
scp traindataold-0000* slave1:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/data
scp traindataold-0000* slave2:/home/worker/xiaoshu/logistic-regression-ftrl-mpi/data
