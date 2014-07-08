#include <iostream>
#include <vector>

#include <thread>
#include <mutex>

#include "../threads.hh"

#define NUM_THREADS 2

std::mutex mtx;

void init_a(int tid,std::vector<int>& a){

    // start of the critical section
    mtx.lock();

    int offset(6*tid);
    for(int i=offset;i!=(6+offset);++i) a[i]=i;

    // output global vector
    std::cout<<"Global vector after the call of init_a from thread "<<tid<<": a={ ";
    for(std::vector<int>::iterator it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
    std::cout<<"}"<<std::endl;

    // end of the critical section
    mtx.unlock();

}

void init_a_optimized(int tid,std::vector<int>& a){

    int offset(6*tid);
    for(int i=offset;i!=(6+offset);++i) a[i]=i;

    // start of the critical section
    mtx.lock();

    // output global vector
    std::cout<<"Global vector after the call of init_a_optimized from thread "<<tid<<": a={ ";
    for(std::vector<int>::iterator it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
    std::cout<<"}"<<std::endl;

    // end of the critical section
    mtx.unlock();

}

void clear_a(int tid,std::vector<int>& a){

    int offset(6*tid);
    for(int i=offset;i!=(6+offset);++i) a[i]=0;

}

void oper_a(int tid,std::vector<int>& a){

    int offset(6*tid);
    for(int i=offset;i!=(6+offset);++i) a[i]+=10*(tid+1);

}

int main(int argc,char** argv){

    // create global vector a
    std::vector<int> a(12,0);

    // launch a group of threads to fill the global vector a with init_a
    std::vector<std::thread> t(NUM_THREADS);

    for(int i=0;i!=NUM_THREADS;++i) t[i]=std::thread(init_a,i,std::ref(a));
    for(int i=0;i!=NUM_THREADS;++i) t[i].join();

    // output global vector a
    std::cout<<"Global vector on main thread: a={ ";
    for(std::vector<int>::iterator it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
    std::cout<<"}"<<std::endl<<std::endl;;

    // clear global vector a
    std::cout<<"Clearing the global vector"<<std::endl<<std::endl;
    for(int i=0;i!=NUM_THREADS;++i) t[i]=std::thread(clear_a,i,std::ref(a));
    for(int i=0;i!=NUM_THREADS;++i) t[i].join();

    // launch a group of threads to fill the global vector a with init_a_optimized
    for(int i=0;i!=NUM_THREADS;++i) t[i]=std::thread(init_a_optimized,i,std::ref(a));
    for(int i=0;i!=NUM_THREADS;++i) t[i].join();

    // output global vector a
    std::cout<<"Global vector on main thread: a={ ";
    for(std::vector<int>::iterator it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
    std::cout<<"}"<<std::endl<<std::endl;

    // do something on global vector a
    std::cout<<"Performing the operation a[i]+=10*(thread_ID+1)"<<std::endl;
    for(int i=0;i!=NUM_THREADS;++i) t[i]=std::thread(oper_a,i,std::ref(a));
    for(int i=0;i!=NUM_THREADS;++i) t[i].join();

    // output global vector a
    std::cout<<"Global vector on main thread: a={ ";
    for(std::vector<int>::iterator it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
    std::cout<<"}"<<std::endl<<std::endl;

    return 0;

}
