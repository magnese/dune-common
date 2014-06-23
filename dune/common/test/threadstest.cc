// 1D finite difference scheme to test threads

#define WORLDDIM 1
#define GRIDDIM 1

#include <iostream>
#include <vector>
#include <algorithm>

#include <dune/common/fvector.hh>

#define DEBUG_THREADS_TEST 0

int main(void){

    // initial description
    std::cout<<"Finite difference scheme over "<<GRIDDIM<<"D grid in a "<<WORLDDIM<<"D world."<<std::endl<<std::endl;

    // geometry input
    typedef Dune::FieldVector<double,WORLDDIM> CoordType;
    CoordType x0;
    CoordType x1;

    std::cout<<"Coordinate of the starting point of the grid : ";
    std::cin>>x0;
    std::cout<<"Coordinate of the ending point of the grid : ";
    std::cin>>x1;
    std::cout<<std::endl;

    // order geometry in the correct way
    if(x0[0]>x1[0]) std::swap(x0[0],x1[0]);

    // input the number of threads
    unsigned int numThreads(0);

    std::cout<<"Number of threads to use : ";
    std::cin>>numThreads;
    if(numThreads<2){
        numThreads=2;
        std::cout<<"Number of threads too small. I am going to use "<<numThreads<<" threads."<<std::endl;
    }

    // input the number of grid elements managed by each thread
    unsigned int numGridElementsPerThread(0);

    std::cout<<"Number of grid elements for each thread : ";
    std::cin>>numGridElementsPerThread;
    std::cout<<std::endl;

    // grid creation
    unsigned int numNodes(numThreads*numGridElementsPerThread+1);
    CoordType x1x0Vector(x1-x0);
    double deltax(x1x0Vector.two_norm()/(numNodes-1));
    std::vector<CoordType> grid(numNodes,x0);

    for(size_t i=1;i!=numNodes;++i) grid[i]+=(deltax*i);

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<std::endl<<"DEBUG: grid coordinates"<<std::endl;
    for(std::vector<CoordType>::iterator it=grid.begin();it!=grid.end();++it) std::cout<<*it<<" ";
    std::cout<<std::endl<<std::endl;
    #endif
    #endif

    // set type of index
    enum flags{shared,NOTshared};
    typedef std::vector<flags> ThreadsIndexType;
    ThreadsIndexType tit(numNodes,NOTshared);

    for(size_t i=0;i!=numThreads;++i){
    }

    return 0;

}
