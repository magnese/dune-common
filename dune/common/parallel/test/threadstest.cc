#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#define NUM_THREADS 2

//global mutex
std::mutex mtx;

template<typename V>
void init(size_t tid,V& v){

  // start of the critical section
  mtx.lock();

  size_t offset(6*tid);
  for(size_t i=offset;i!=(6+offset);++i) v[i]=i;

  // output global vector
  std::cout<<"[thread "<<tid<<"] vector after init(): ";
  for(typename V::iterator it=v.begin();it!=v.end();++it) std::cout<<*it<<" ";
  std::cout<<std::endl;

  // end of the critical section
  mtx.unlock();

}

template<typename V>
void initOptimized(size_t tid,V& v){

  size_t offset(6*tid);
  for(size_t i=offset;i!=(6+offset);++i) v[i]=i;

  // start of the critical section
  mtx.lock();

  // output global vector
  std::cout<<"[thread "<<tid<<"] vector after initOptimized(): ";
  for(typename V::iterator it=v.begin();it!=v.end();++it) std::cout<<*it<<" ";
  std::cout<<std::endl;

  // end of the critical section
  mtx.unlock();

}

template<typename V>
void clear(size_t tid,V& v){
  size_t offset(6*tid);
  for(size_t i=offset;i!=(6+offset);++i) v[i]=0;
}

template<typename V>
void oper(size_t tid,V& v){
  size_t offset(6*tid);
  for(size_t i=offset;i!=(6+offset);++i) v[i]+=10*(tid+1);
}

int main(int argc,char** argv){

  // create vector a
  typedef int ctype;
  typedef typename std::vector<ctype> VectorType;
  VectorType a(12,0);
  typedef typename VectorType::iterator VectorItType;

  // launch a group of threads to fill a with init()
  const size_t numThreads(NUM_THREADS);
  std::vector<std::thread> t(numThreads);

  for(size_t i=0;i!=numThreads;++i) t[i]=std::thread(init<VectorType>,i,std::ref(a));
  for(size_t i=0;i!=numThreads;++i) t[i].join();

  // output a
  std::cout<<"[main] vector: ";
  for(VectorItType it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
  std::cout<<std::endl<<std::endl;;

  // clear a with clear()
  std::cout<<"Clearing the vector"<<std::endl<<std::endl;
  for(size_t i=0;i!=numThreads;++i) t[i]=std::thread(clear<VectorType>,i,std::ref(a));
  for(size_t i=0;i!=numThreads;++i) t[i].join();

  // launch a group of threads to fill a with initOptimized()
  for(size_t i=0;i!=numThreads;++i) t[i]=std::thread(initOptimized<VectorType>,i,std::ref(a));
  for(size_t i=0;i!=numThreads;++i) t[i].join();

  // output a
  std::cout<<"[main] vector: ";
  for(VectorItType it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
  std::cout<<std::endl<<std::endl;

  // do something a with oper()
  std::cout<<"Performing the operation v[i]+=10*(ThreadID+1)"<<std::endl<<std::endl;
  for(size_t i=0;i!=numThreads;++i) t[i]=std::thread(oper<VectorType>,i,std::ref(a));
  for(size_t i=0;i!=numThreads;++i) t[i].join();

  // output a
  std::cout<<"[main] vector: ";
  for(VectorItType it=a.begin();it!=a.end();++it) std::cout<<*it<<" ";
  std::cout<<std::endl<<std::endl;

  return 0;

}
