// 1D FEM scheme for equation -u''=f with Dirichlet boundary conditions to test threads

#define WORLDDIM 1
#define GRIDDIM 1

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

#include <dune/common/fvector.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>

// problem definition
// exact solution x^2+1
#define COOR_X0 0
#define COOR_X1 1
inline double f(double&& x){return 2.0;}
double ux0(1.0);
double ux1(2.0);

// threads control
#define NUM_THREADS 4
#define GRID_ELEMENTS_PER_THREAD 2
#define DEBUG_THREADS_TEST 1

// basis functions
inline double phi0(double&& x){return 1.0-x;}
inline double phi1(double&& x){return x;}
inline double derphi0(double&& x){return -1.0;}
inline double derphi1(double&& x){return 1.0;}

// assemble function
template<typename AType,typename bType,typename IType,typename GType,typename FType>
void assemble(unsigned int tid,AType& A,bType& b,IType& idx,GType& grid,unsigned int numGridElements,FType& phi,FType& derphi){

    std::vector<std::vector<double>> Alocal(2,std::vector<double>(2,0.0));
    std::vector<double> blocal(2,0.0);

    unsigned int startElem(tid*numGridElements);
    unsigned int endElem((tid+1)*numGridElements);
    for(unsigned int elem=startElem;elem!=endElem;++elem){

        // assemble local A and local b
        for(unsigned int i=0;i!=2;++i){

            blocal[i]=0.0;
            // using trapezoid rule
            blocal[i]+=0.5*(phi[i](0.0)*f(0.0));
            blocal[i]+=0.5*(phi[i](1.0)*f(1.0));
            blocal[i]*=(grid[elem+1]-grid[elem]);

            for(unsigned int j=0;j!=2;++j){
                Alocal[i][j]=0.0;
                // using trapezoid rule
                Alocal[i][j]+=0.5*(derphi[i](0.0)*derphi[j](0.0));
                Alocal[i][j]+=0.5*(derphi[i](1.0)*derphi[j](1.0));
                Alocal[i][j]/=(grid[elem+1]-grid[elem]);
            }

        }

        // adding local A and local b to the global stiffness matrix and the global RHS
        for(unsigned int i=0;i!=2;++i){
            b[elem+i]+=blocal[i];
            for(unsigned int j=0;j!=2;++j){
                A[elem+i][elem+j]+=Alocal[i][j];
            }
        }

    }

}


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

    // basis functions
    typedef std::function<double(double&&)> FunctionType;

    std::vector<FunctionType> phi(2);
    phi[0]=phi0; // phi[0]([](double& x)->double{return 1.0-x;});
    phi[1]=phi1; // phi[1]([](double& x)->double{return x;});

    std::vector<FunctionType> derphi(2);
    derphi[0]=derphi0; // derphi[0]([](double& x)->double{return -1.0;});
    derphi[1]=derphi1; // derphi[1]([](double& x)->double{return 1;});

    // assemble stiffness matrix and RHS
    for(unsigned int i=0;i!=numThreads;++i) assemble(i,A,b,tit,grid,numGridElementsPerThread,phi,derphi);

    // impose boundary condition
    b[0]=ux0;
    b[numNodes-1]=ux1;


    return 0;

}
