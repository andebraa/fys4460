#include <iostream>
#include <armadillo>
//#include <mpi.h>
//#include <mpi/mpi.h>
#define ZERO  1.0E-10
#include "heather.h"
#include <fstream>
#include <time.h>

using namespace std;

int main(){

  ifstream myfile;
  myfile.open("dump.lammps_0.5")
  if (myfile.is_open()){
    for(int i =0; i < 100 & myfile.eof(); i++){
      getline(myfile, line);
      if (line[:14] = "ITEM: TIMESTEP"){
        break;
      }
    }

  }

}
