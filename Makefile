#!/bin/bash
INCLUDEPATH = -I/usr/local/include/ 
LIBRARYPATH = -L/usr/local/lib 
LIBRARY = -lpthread -lglog
#train code
CPP_tag = -std=gnu++11

train:main.o
	mpicxx $(CPP_tag) -o train main.o $(LIBRARYPATH) $(LIBRARY)

main.o: src/main.cpp
	mpicxx $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp

train_ut: train_uttest.o owlqn.o
	mpicxx -o train_ut train_uttest.o owlqn.o $(LIB) $(GLOG_LIB) -lopenblas -lpthread -L ./lib -lgtest

train_uttest.o: src/train_uttest.cpp
	mpicxx  -I ./include $(GLOG_INCLUDE) -c src/train_uttest.cpp

clean:
	rm -f *~ train predict train_ut *.o
