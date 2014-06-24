// 1D FEM scheme for equation -u''=f with Dirichlet boundary conditions to test threads

#define WORLDDIM 1
#define GRIDDIM 1

#include <iostream>
#include <vector>
#include <algorithm>

#include <dune/common/fvector.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>

// problem definition
// exact solution x^2+1
#define COOR_X0 0
#define COOR_X1 1
inline double f(double& x){return 2.0;}
double ux0(1.0);
double ux1(2.0);

// threads control
#define NUM_THREADS 4
#define GRID_ELEMENTS_PER_THREAD 2
#define DEBUG_THREADS_TEST 1

// basis functions
inline double phi0(double& x){return 1.0-x;}
inline double phi1(double& x){return x;}

int main(void){

    // initial description
    std::cout<<"FEM scheme over "<<GRIDDIM<<"D grid in a "<<WORLDDIM<<"D world to solve equation -u''=f with Dirichlet boundary conditions u("<<COOR_X0<<")="<<ux0<<" and u("<<COOR_X1<<")="<<ux1<<"."<<std::endl<<std::endl;

    // geometry definition
    typedef Dune::FieldVector<double,WORLDDIM> CoordType;
    CoordType x0(COOR_X0);
    CoordType x1(COOR_X1);

    if(x0[0]>x1[0]){
        std::swap(x0[0],x1[0]);
        std::swap(ux0,ux1);
    }

    std::cout<<"Coordinate of the starting point of the grid : "<<x0<<std::endl;
    std::cout<<"Coordinate of the ending point of the grid : "<<x1<<std::cout<<std::endl;

    // number of threads
    unsigned int numThreads((NUM_THREADS<2?2:NUM_THREADS));
    std::cout<<"Number of threads to use : "<<numThreads<<std::endl;

    // number of grid elements managed by each thread
    unsigned int numGridElementsPerThread((GRID_ELEMENTS_PER_THREAD<1?1:GRID_ELEMENTS_PER_THREAD));
    std::cout<<"Number of grid elements for each thread : "<<numGridElementsPerThread<<std::endl<<std::endl;

    // grid creation
    unsigned int numNodes(numThreads*numGridElementsPerThread+1);
    CoordType x1x0Vector(x1-x0);
    double deltax(x1x0Vector.two_norm()/(numNodes-1));
    std::vector<CoordType> grid(numNodes,x0);

    for(size_t i=1;i!=numNodes;++i) grid[i]+=(deltax*i);

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: grid coordinates"<<std::endl;
    for(std::vector<CoordType>::iterator it=grid.begin();it!=grid.end();++it) std::cout<<*it<<" ";
    std::cout<<std::endl<<std::endl;
    #endif
    #endif

    // set type of index
    enum flags{shared,NOTshared};
    typedef std::vector<flags> ThreadsIndexType;
    ThreadsIndexType tit(numNodes,NOTshared);

    for(size_t i=0;i!=(numThreads-1);++i){
        tit[(i+1)*numGridElementsPerThread]=shared;
    }

    #ifdef DEBUG_THREADS_TEST
    #if DEBUG_THREADS_TEST
    std::cout<<"DEBUG: shared-NOTshared nodes"<<std::endl;
    for(ThreadsIndexType::iterator it=tit.begin();it!=(tit.end()-1);++it){
        if(*it==NOTshared) std::cout<<"N--";
        if(*it==shared) std::cout<<"S--";
    }
    if(*(tit.rbegin())==NOTshared) std::cout<<"N"<<std::endl<<std::endl;
    if(*(tit.rbegin())==shared) std::cout<<"S"<<std::endl<<std::endl;
    #endif
    #endif

    // allocate stiffness matrix A, RHS vector b and solution vector x
    typedef Dune::DynamicMatrix<double> StiffnessMatrixType;
    StiffnessMatrixType A(numNodes,numNodes,0.0);

    typedef Dune::DynamicVector<double> VectorType;
    VectorType b(numNodes,0.0);
    VectorType x(numNodes,0.0);

    //values first derivative basis function
    double derphi0(-1.0);
    double derphi1(1.0);

    // assemble stiffness matrix and RHS


    return 0;

}
