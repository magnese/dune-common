#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>

#include <dune/common/enumset.hh>

#include <dune/common/parallel/mpicollectivecommunication.hh>
#include <dune/common/parallel/indexset.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/common/parallel/remoteindices.hh>
#include <dune/common/parallel/interface.hh>
#include <dune/common/parallel/communicator.hh>
#include <dune/common/parallel/parallelparadigm.hh>

// policy: copy
template<typename T>
class CopyData{

public:

  typedef typename T::value_type IndexedType;

  static IndexedType gather(const T& v,int i){return v[i];}
  static void scatter(T& v,IndexedType item,int i){v[i]=item;}

};

// policy: add
template<typename T>
class AddData{

public:

  typedef typename T::value_type IndexedType;

  static IndexedType gather(const T& v,int i){return v[i];}
  static void scatter(T& v,IndexedType item,int i){v[i]+=item;}

};


int main(int argc,char** argv){

  #if HAVE_MPI

  // init MPI
  MPI_Init(&argc,&argv);
  Dune::CollectiveCommunication<MPI_Comm> collCom(MPI_COMM_WORLD);

  // get size and rank
  size_t size(collCom.size());
  size_t rank(collCom.rank());

  // 2 processes
  if(size>2){
    if(rank==0) std::cout<<std::endl<<"WARNING: run me again with just 2 processes"<<std::endl;
  }
  else{

    // define parallel local index and parallel index set
    enum flags{owner,ghost};
    typedef Dune::ParallelLocalIndex<flags> LocalIndexType;
    typedef Dune::ParallelIndexSet<size_t,LocalIndexType,7> ParallelIndexType;

    // create parallel index set s
    ParallelIndexType sis;
    size_t offsetGlobal(0);
    size_t offsetLocal(0);

    if(rank==1){
      offsetGlobal=6;
      offsetLocal=1;
    }

    sis.beginResize();
    for(size_t i=0;i!=6;++i) sis.add(i+offsetGlobal,LocalIndexType(i+offsetLocal,owner));
    if(rank==0) sis.add(6,LocalIndexType(6,ghost));
    else sis.add(5,LocalIndexType(0,ghost));
    sis.endResize();

    // output sis
    for(size_t i=0;i!=size;++i){
      if(rank==i){
        std::cout<<"s"<<rank<<":"<<std::endl;
        std::cout<<sis<<std::endl<<std::endl;
      }
      collCom.barrier();
    }

    // create parallel paradigm
    typedef Dune::MPIParadigm<ParallelIndexType> ParallelParadigmType;
    ParallelParadigmType pp(MPI_COMM_WORLD);

    // set remote indices
    typedef Dune::RemoteIndices<ParallelIndexType> RemoteIndicesType;
    RemoteIndicesType riS(sis,sis,MPI_COMM_WORLD);
    riS.rebuild<true>();

    // create interface
    Dune::EnumItem<flags,ghost> ghostFlags;
    Dune::EnumItem<flags,owner> ownerFlags;

    typedef Dune::Interface InterfaceType;
    InterfaceType infS(MPI_COMM_WORLD);
    infS.build(riS,ownerFlags,ghostFlags);

    // create local vector al
    typedef int ctype;
    typedef typename std::vector<ctype> VectorType;
    VectorType al(7,0);
    typedef typename VectorType::iterator VectorItType;

    typedef typename ParallelIndexType::iterator PIndexIterType;
    for(PIndexIterType it=sis.begin();it!=sis.end();++it){
      if(it->local().attribute()==owner) al[it->local().local()]=(it-sis.begin())+5*rank;
    }

    // output al
    for(size_t i=0;i!=size;++i){
      if(rank==i){
        std::cout<<"Local vector on process "<<rank<<": al={ ";
        for(VectorItType it=al.begin();it!=al.end();++it) std::cout<<*it<<" ";
        std::cout<<"}"<<std::endl;
      }
      collCom.barrier();
    }

    // do something on al
    if(rank==0) std::cout<<std::endl<<"Performing the operation al[i]+=10*(rank+1) for only the owned entries"<<std::endl<<std::endl;
    for(PIndexIterType it=sis.begin();it!=sis.end();++it){
      if(it->local().attribute()==owner) al[it->local().local()]+=10*(rank+1);
    }

    // output al before communication
    for(size_t i=0;i!=size;++i){
      if(rank==i){
        std::cout<<"Local vector on process "<<rank<<": al={ ";
        for(VectorItType it=al.begin();it!=al.end();++it) std::cout<<*it<<" ";
        std::cout<<"}"<<std::endl;
      }
      collCom.barrier();
    }

    // create communicator
    typedef Dune::BufferedCommunicator CommunicatorType;
    CommunicatorType bComm;
    bComm.build(al,al,infS);

    // communicate
    if(rank==0) std::cout<<std::endl<<"Forward communication"<<std::endl<<std::endl;
    bComm.forward<CopyData<VectorType>>(al,al);

    // output al after communication
    for(size_t i=0;i!=size;++i){
      if(rank==i){
        std::cout<<"Local vector on process "<<rank<<": al={ ";
        for(VectorItType it=al.begin();it!=al.end();++it) std::cout<<*it<<" ";
        std::cout<<"}"<<std::endl;
      }
      collCom.barrier();
    }

  }

  // finalize MPI
  MPI_Finalize();

  #endif //HAVE_MPI

  return 0;

}
