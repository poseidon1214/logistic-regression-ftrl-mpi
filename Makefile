#!/bin/bash
LIB = -L/opt/OpenBLAS/lib
INCLUDE = -I/opt/OpenBLAS/include 
GLOG_LIB = -L/usr/local/lib
GLOG_INCLUDE = -I/usr/local/include/glog
GTEST_LIB = -L/usr/local/lib
GTEST_INCLUDE = -I/usr/local/include/gtest
#train code
CPP_tag = -std=gnu++11

train:lr_main.o
	mpicxx $(CPP_tag) -o train lr_main.o $(INCLUDE) $(LIB) -lpthread -lopenblas -lglog

lr_main.o: src/lr_main.cpp
	mpicxx $(CPP_tag) $(INCLUDE) $(GLOG_INCLUDE) -c src/lr_main.cpp

#predict code
#predict: predict.o
#	mpicxx -g -o predict -lpthread $(LIB) -lopenblas predict.o
#predict.o: src/predict.cpp
#	mpicxx $(INCLUDE) -c src/predict.cpp

#make train uttest
train_ut: train_uttest.o owlqn.o
	mpicxx -o train_ut train_uttest.o owlqn.o $(LIB) $(GLOG_LIB) -lopenblas -lpthread -L ./lib -lgtest

train_uttest.o: src/train_uttest.cpp
	mpicxx  -I ./include $(GLOG_INCLUDE) -c src/train_uttest.cpp

clean:
	rm -f *~ train predict train_ut *.o
