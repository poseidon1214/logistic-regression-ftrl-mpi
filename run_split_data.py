python split_data.py train_data.txt 3 train_new
scp test_new-0000* train_new-0000* slave1:/home/worker/xiaoshu/DML/logistic_regression_mpi/data
scp test_new-0000* train_new-0000* slave2:/home/worker/xiaoshu/DML/logistic_regression_mpi/data
