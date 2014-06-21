#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>
#include <thread>

#define NUM_THREADS 2

void init_a(int tid,std::vector<int>& a){

    if(tid==0){
        for(int i=0;i!=6;++i) a[i]=i;
    }

    if(tid==1){
        for(int i=6;i!=12;++i) a[i]=i;
    }

}

int main(int argc,char** argv){

    // create global vector a
    std::vector<int> a(12,0);

    //launch a group of threads to fill the global vector
    std::thread t[NUM_THREADS];

    for(int i=0;i!=NUM_THREADS;++i){
        t[i]=std::thread(init_a,i,a);
    }

    for(int i=0;i!=NUM_THREADS;++i) t[i].join();

    // output global vector
    std::cout<<"Global vector on main thread: a={ ";
    for(std::vector<int>::iterator it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
    std::cout<<"}"<<std::endl;
/*
        // do something on al
        if(rank==0) std::cout<<std::endl<<"Performing the operation al[i]+=10*(rank+1) for only the owned entries"<<std::endl<<std::endl;
        for(PIndexIterType it=sis.begin();it!=sis.end();++it){
            if(it->local().attribute()==owner) al[it->local().local()]+=10*(rank+1);
        }

        // output al before communication
        for(int i=0;i!=nProcs;++i){
            if(rank==i){
                std::cout<<"Local vector on process "<<rank<<": al={ ";
                for(std::vector<int>::iterator it=al.begin();it!=al.end();++it) std::cout<<*it<<" ";
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
        bComm.forward<CopyData<std::vector<int> > >(al,al);

        // output al after communication
        for(int i=0;i!=nProcs;++i){
            if(rank==i){
                std::cout<<"Local vector on process "<<rank<<": al={ ";
                for(std::vector<int>::iterator it=al.begin();it!=al.end();++it) std::cout<<*it<<" ";
                std::cout<<"}"<<std::endl;
            }
            collCom.barrier();
        }

    }

    // finalize MPI
    MPI_Finalize();

    #endif //HAVE_MPI
*/
    return 0;

}
