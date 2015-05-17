// 1D FEM scheme for equation -u''=f with Dirichlet boundary conditions to test threads

#define WORLDDIM 1
#define GRIDDIM 1

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

#include <thread>
#include <mutex>

#include <dune/common/fvector.hh>
#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>

// problem definition
// exact solution x^2+1
#define COOR_START 0.0
#define COOR_END 1.0
inline double f(const double& x)
{
  return 2.0;
}
double ux0(1.0);
double ux1(2.0);

// threads control
#define NUM_THREADS 4
#define GRID_ELEMENTS_PER_THREAD 2
#define DEBUG_FLAG 1

// basis functions
inline double phi0(const double& x)
{
  return 1.0-x;
}
inline double phi1(const double& x)
{
  return x;
}
inline double derphi0(const double& x)
{
  return -1.0;
}
inline double derphi1(const double& x)
{
  return 1.0;
}

// flag
enum flags{shared,NOTshared};

// assemble function
template<typename AType,typename bType,typename IType,typename GType,typename FType>
void assemble(unsigned int tid,AType& A,bType& b,IType& idx,GType& grid,unsigned int numGridElements,FType& phi,FType& derphi)
{
  std::vector<std::vector<double>> Alocal(2,std::vector<double>(2,0.0));
  std::vector<double> blocal(2,0.0);

  std::size_t startElem(tid*numGridElements);
  std::size_t endElem((tid+1)*numGridElements);
  for(std::size_t elem=startElem;elem!=endElem;++elem)
  {
    // assemble local A and local b
    for(std::size_t i=0;i!=2;++i)
    {
      blocal[i]=0.0;
      // using trapezoid rule
      blocal[i]+=0.5*(phi[i](0.0)*f(0.0));
      blocal[i]+=0.5*(phi[i](1.0)*f(1.0));
      blocal[i]*=(grid[elem+1]-grid[elem]);

      for(std::size_t j=0;j!=2;++j)
      {
        Alocal[i][j]=0.0;
        // using trapezoid rule
        Alocal[i][j]+=0.5*(derphi[i](0.0)*derphi[j](0.0));
        Alocal[i][j]+=0.5*(derphi[i](1.0)*derphi[j](1.0));
        Alocal[i][j]/=(grid[elem+1]-grid[elem]);
      }
    }
    // adding local A and local b to the global stiffness matrix and the global RHS
    for(std::size_t i=0;i!=2;++i)
    {
      // possible critical section
      if(idx[elem+i].first==shared)
        idx[elem+i].second->lock();
      b[elem+i]+=blocal[i];
      for(std::size_t j=0;j!=2;++j)
      {
        // possible critical section
        if(idx[elem+j].first==shared)
          idx[elem+j].second->lock();
        A[elem+i][elem+j]+=Alocal[i][j];
        if(idx[elem+j].first==shared)
          idx[elem+j].second->unlock();
      }
      if(idx[elem+i].first==shared)
        idx[elem+i].second->unlock();
    }
  }

}

// print string
inline void print(const std::string& str)
{
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  std::cout<<"DEBUG: "<<str<<std::endl;
  #endif
  #endif
}

// print vector function
template<typename V>
inline void printVector(const V& v)
{
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  for(auto val:v)
    std::cout<<val<<" ";
  std::cout<<std::endl;
  #endif
  #endif
}

// print matrix function
template<typename M>
inline void printMatrix(const M& m)
{
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  for(auto row:m)
    printVector(row);
  std::cout<<std::endl;
  #endif
  #endif
}

int main(int argc, char** argv)
{
  // initial description
  std::cout<<"FEM scheme over "<<GRIDDIM<<"D grid in a "<<WORLDDIM<<"D world to solve equation -u''=f with Dirichlet boundary conditions u("
    <<COOR_START<<")="<<ux0<<" and u("<<COOR_END<<")="<<ux1<<"."<<std::endl<<std::endl;

  // geometry definition
  typedef double ctype;
  typedef Dune::FieldVector<ctype,WORLDDIM> CoordType;
  CoordType x0(COOR_START);
  CoordType x1(COOR_END);

  if(x0[0]>x1[0])
  {
    std::swap(x0[0],x1[0]);
    std::swap(ux0,ux1);
  }

  std::cout<<"Coordinate of the starting point of the grid : "<<x0<<std::endl;
  std::cout<<"Coordinate of the ending point of the grid : "<<x1<<std::cout<<std::endl;

  // number of threads
  const std::size_t numThreads((NUM_THREADS<2?2:NUM_THREADS));
  std::cout<<"Number of threads to use : "<<numThreads<<std::endl;

  // number of grid elements managed by each thread
  const std::size_t numGridElementsPerThread((GRID_ELEMENTS_PER_THREAD<1?1:GRID_ELEMENTS_PER_THREAD));
  std::cout<<"Number of grid elements for each thread : "<<numGridElementsPerThread<<std::endl<<std::endl;

  // grid creation
  const std::size_t numNodes(numThreads*numGridElementsPerThread+1);
  CoordType x1x0Vector(x1-x0);
  ctype deltax(x1x0Vector.two_norm()/(numNodes-1));
  typedef std::vector<CoordType> GridType;
  GridType grid(numNodes,x0);

  for(std::size_t i=1;i!=numNodes;++i)
    grid[i]+=(deltax*i);

  print("grid coordinates");
  printVector(grid);

  // set type of index (one mutex for each shared part of the vector)
  typedef std::vector<std::pair<flags,std::recursive_mutex*>> ThreadsIndexType;
  ThreadsIndexType tit(numNodes,std::pair<flags,std::recursive_mutex*>(NOTshared,nullptr));

  for(std::size_t i=0;i!=(numThreads-1);++i)
  {
    tit[(i+1)*numGridElementsPerThread].first=shared;
    tit[(i+1)*numGridElementsPerThread].second=new std::recursive_mutex();
  }

  print("shared-NOTshared nodes");
  #ifdef DEBUG_FLAG
  #if DEBUG_FLAG
  for(ThreadsIndexType::iterator it=tit.begin();it!=(tit.end()-1);++it)
  {
    if(it->first==NOTshared)
      std::cout<<"N--";
    if(it->first==shared)
      std::cout<<"S--";
  }
  if(tit.back().first==NOTshared)
    std::cout<<"N"<<std::endl<<std::endl;
  if(tit.back().first==shared)
    std::cout<<"S"<<std::endl<<std::endl;
  #endif
  #endif

  // allocate stiffness matrix A, RHS vector b and solution vector x
  typedef Dune::DynamicMatrix<ctype> StiffnessMatrixType;
  StiffnessMatrixType A(numNodes,numNodes,0.0);

  typedef Dune::DynamicVector<ctype> VectorType;
  VectorType b(numNodes,0.0);
  VectorType x(numNodes,0.0);

  // basis functions
  typedef std::function<double(const double&)> FunctionType;

  std::vector<FunctionType> phi(2);
  phi[0]=phi0;
  phi[1]=phi1;

  std::vector<FunctionType> derphi(2);
  derphi[0]=derphi0;
  derphi[1]=derphi1;

  // launch a group of threads to assemble the stiffness matrix and the RHS
  std::vector<std::thread> t(numThreads);

  for(std::size_t i=0;i!=numThreads;++i)
    t[i]=std::thread(assemble<StiffnessMatrixType,VectorType,ThreadsIndexType,GridType,std::vector<FunctionType>>,i,std::ref(A),std::ref(b),std::ref(tit),
                      std::ref(grid),numGridElementsPerThread,std::ref(phi),std::ref(derphi));
  for(std::size_t i=0;i!=numThreads;++i)
    t[i].join();

  // impose boundary condition
  A[0][0]=1;
  for(std::size_t i=1;i!=numNodes;++i)
    A[0][i]=0;
  b[0]=ux0;

  A[numNodes-1][numNodes-1]=1;
  for(std::size_t i=0;i!=(numNodes-1);++i)
    A[numNodes-1][i]=0;
  b[numNodes-1]=ux1;

  print("global stiffness matrix");
  printMatrix(A);
  print("global RHS");
  printVector(b);

  // solve the problem
  A.solve(x,b);

  print("global solution");
  printVector(x);

  // clean mutex
  for(auto thrIdx:tit)
  {
    if(thrIdx.first==shared)
      delete thrIdx.second;
  }

  return 0;
}
