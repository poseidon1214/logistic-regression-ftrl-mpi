#!/bin/bash
INCLUDEPATH = -I/usr/local/include/ -I/usr/include
LIBRARYPATH = -L/usr/local/lib 
#LIBRARY = -lboost_thread -lboost_system -lpthread -lglog -lm
LIBRARY = -lpthread -lm
CPP_tag = -std=gnu++11

train:main.o
	mpicxx $(CPP_tag) -o train main.o $(LIBRARYPATH) $(LIBRARY)

main.o: src/main.cpp
	mpicxx $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp

clean:
	rm -f *~ train predict train_ut *.o
