#if HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

//global mutex
std::mutex mtx;

template<typename V>
void init(std::size_t tid,V& v)
{
  // start of the critical section
  mtx.lock();

  std::size_t offset(6*tid);
  for(std::size_t i=offset;i!=(6+offset);++i)
    v[i]=i;

  // output global vector
  std::cout<<"[thread "<<tid<<"] vector after init(): ";
  for(auto val:v)
    std::cout<<val<<" ";
  std::cout<<std::endl;

  // end of the critical section
  mtx.unlock();
}

template<typename V>
void initOptimized(std::size_t tid,V& v)
{
  std::size_t offset(6*tid);
  for(std::size_t i=offset;i!=(6+offset);++i)
    v[i]=i;

  // start of the critical section
  mtx.lock();

  // output global vector
  std::cout<<"[thread "<<tid<<"] vector after initOptimized(): ";
  for(auto val:v)
    std::cout<<val<<" ";
  std::cout<<std::endl;

  // end of the critical section
  mtx.unlock();
}

template<typename V>
void clear(std::size_t tid,V& v)
{
  std::size_t offset(6*tid);
  for(std::size_t i=offset;i!=(6+offset);++i)
    v[i]=0;
}

template<typename V>
void oper(std::size_t tid,V& v)
{
  std::size_t offset(6*tid);
  for(std::size_t i=offset;i!=(6+offset);++i)
    v[i]+=10*(tid+1);
}

int main(int argc,char** argv)
{
  // create vector a
  typedef typename std::vector<int> VectorType;
  VectorType a(12,0);

  // launch a group of threads to fill a with init()
  const std::size_t numThreads(2);
  std::vector<std::thread> t(numThreads);

  for(std::size_t i=0;i!=numThreads;++i)
    t[i]=std::thread(init<VectorType>,i,std::ref(a));
  for(std::size_t i=0;i!=numThreads;++i)
    t[i].join();

  // output a
  std::cout<<"[main] vector: ";
  for(auto val:a)
    std::cout<<val<<" ";
  std::cout<<std::endl<<std::endl;;

  // clear a with clear()
  std::cout<<"Clearing the vector"<<std::endl<<std::endl;
  for(std::size_t i=0;i!=numThreads;++i)
    t[i]=std::thread(clear<VectorType>,i,std::ref(a));
  for(std::size_t i=0;i!=numThreads;++i)
    t[i].join();

  // launch a group of threads to fill a with initOptimized()
  for(std::size_t i=0;i!=numThreads;++i)
    t[i]=std::thread(initOptimized<VectorType>,i,std::ref(a));
  for(std::size_t i=0;i!=numThreads;++i)
    t[i].join();

  // output a
  std::cout<<"[main] vector: ";
  for(auto val:a)
    std::cout<<val<<" ";
  std::cout<<std::endl<<std::endl;

  // do something a with oper()
  std::cout<<"Performing the operation v[i]+=10*(ThreadID+1)"<<std::endl<<std::endl;
  for(std::size_t i=0;i!=numThreads;++i)
    t[i]=std::thread(oper<VectorType>,i,std::ref(a));
  for(std::size_t i=0;i!=numThreads;++i)
    t[i].join();

  // output a
  std::cout<<"[main] vector: ";
  for(auto val:a)
    std::cout<<val<<" ";
  std::cout<<std::endl<<std::endl;

  return 0;
}
