// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADPARALLELPARADIGM_HH
#define DUNE_THREADPARALLELPARADIGM_HH

#include "indexset.hh"
#include "plocalindex.hh"
#include <dune/common/stdstreams.hh>
#include <utility>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>
//#include <thread>
#include <mutex>
#include <condition_variable>

namespace Dune {
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Class implementing the thread parallel paradigm.
   * @author Marco Agnese, Markus Blatt
   */

  /** @brief ThreadCommunicator allows communication between threads using shared memeory. */
  template<size_t numThreads>
  class ThreadCommunicator
  {
    public:

      /** @brief Default constructor. */
      inline ThreadCommunicator();

      /** @brief Destructor. */
      inline ~ThreadCommunicator()
      {}

      /** @brief Get the number of threads. */
      inline const size_t& size() const;

      /** @brief Barrier to syncronise threads. */
      inline void barrier();

      /** @brief Create buffer to share data among threads. */
      template<class T>
      inline void createBuffer(const size_t& tid);

      /** @brief Set a value into the buffer. */
      template<class T>
      inline void setBuffer(const T& value, const size_t& tid);

      /** @brief Get buffer. */
      template<class T>
      inline std::array<T,numThreads>& getBuffer() const;

      /** @brief Delete buffer. */
      template<class T>
      inline void deleteBuffer(const size_t& tid);

    private:
      /** @brief Number of threads. */
      const size_t size_;

      /** @brief Pointer to the buffer. */
      void* bufferptr_;

      /** @brief Mutex used for the barrier. */
      std::mutex mtx_;

      /** @brief Counter used for the barrier. */
      size_t count_;

      /** @brief Second counter used for the barrier. */
      size_t seccount_;

      /** @brief Condition variable used for the barrier. */
      std::condition_variable condvar_;

  };

  /**
   * @brief ThreadParadigm.
   * @tparam T The type of the underlying index set.
   * @tparam C The type of the communicator.
   */
  template<class T, class C>
  class ThreadParadigm
  {

  public:
    /** @brief Type of the index set we use, e.g. ParallelLocalIndexSet. */
    typedef T ParallelIndexSet;

    /** @brief The type of the global index. */
    typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

    /** @brief The type of the local index. */
    typedef typename ParallelIndexSet::LocalIndex LocalIndex;

    /** @brief The type of the attribute. */
    typedef typename LocalIndex::Attribute Attribute;

    /** @brief The type of the communicator. */
    typedef C CommType;

    /** @brief Constructor. */
    inline ThreadParadigm(CommType& comm, const size_t& tid);

    /** @brief Default constructor. */
    ThreadParadigm()
    {}

     /** @brief Set the paradigm we work with. */
    inline void setParadigm(CommType& comm, const size_t& tid);

    /** @brief Destructor. */
    ~ThreadParadigm()
    {}

    /** @brief Get the thread ID. */
    inline size_t threadID() const;

   /** @brief Get the communicator. */
    inline const CommType& communicator() const;

    //! \todo Please finsih to doc me.
    /**
     * @brief Build the remote mapping. If the template parameter ignorePublic is true all indices will be treated as public.
     * @param includeSelf If true, sending from indices of the processor to other indices on the same processor is enabled even
     * if the same indexset is used on both the sending and receiving side.
     */
    template<bool ignorePublic, class RemoteIndexList, class RemoteIndexMap = std::map<int, std::pair<RemoteIndexList*,RemoteIndexList*> > >
    inline void buildRemote(const ParallelIndexSet* source, const ParallelIndexSet* target, RemoteIndexMap& remoteIndices,
                            std::set<int>& neighbourIds, bool includeSelf);

  private:
    /** copying is forbidden. */
    ThreadParadigm(const ThreadParadigm&)
    {}

    /** @brief The communicator. */
    CommType& comm_;

    /** @brief The thread ID. */
    size_t tid_;

    /** @brief The index pair type. */
    typedef IndexPair<GlobalIndex,LocalIndex> PairType;

  };

  /** @} */

  template<size_t numThreads>
  inline ThreadCommunicator<numThreads>::ThreadCommunicator() : size_(numThreads), bufferptr_(nullptr), mtx_(), count_(0), seccount_(0), condvar_()
  {}

  template<size_t numThreads>
  inline const size_t& ThreadCommunicator<numThreads>::size() const
  {
    return size_;
  }

  template<size_t numThreads>
  inline void ThreadCommunicator<numThreads>::barrier()
  {
    std::unique_lock<std::mutex> lock(mtx_);
    const size_t oldSecCount = seccount_;
    ++count_;
    if(count_ != size_)
    {
      // wait condition must not depend on wait_count
      condvar_.wait(lock, [this, oldSecCount]() {return seccount_ != oldSecCount;});
    }
    else
    {
      count_ = 0; // reset counter for later use of the barrier //TODO: fix the reset, may lead to a segmentation fault
      // increasing the second counter allows waiting threads to exit
      ++seccount_;
      condvar_.notify_all();
    }
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::createBuffer(const size_t& tid)
  {
    if(tid==0)
      bufferptr_ = new std::array<T,numThreads>();
    barrier(); // syncronization
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::setBuffer(const T& value, const size_t& tid)
  {
    (*(static_cast<std::array<T,numThreads>*>(bufferptr_)))[tid] = value;
    barrier(); // syncronization
  }

  template<size_t numThreads>
  template<typename T>
  inline std::array<T,numThreads>& ThreadCommunicator<numThreads>::getBuffer() const
  {
    return *(static_cast<std::array<T,numThreads>*>(bufferptr_));
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::deleteBuffer(const size_t& tid)
  {
    if(tid==0)
    {
      delete static_cast<std::array<T,numThreads>*>(bufferptr_);
      bufferptr_ = nullptr;
    }
    barrier(); // syncronization //TODO: maybe useless
  }

  template<typename T,typename C>
  inline ThreadParadigm<T,C>::ThreadParadigm(CommType& comm, const size_t& tid) : comm_(comm), tid_(tid)
  {}

  template<typename T,typename C>
  inline void ThreadParadigm<T,C>::setParadigm(CommType& comm, const size_t& tid)
  {
    comm_ = comm;
    tid_ = tid;
  }

  template<typename T,typename C>
  inline size_t ThreadParadigm<T,C>::threadID() const
  {
    return tid_;
  }

  template<typename T,typename C>
  inline const typename ThreadParadigm<T,C>::CommType& ThreadParadigm<T,C>::communicator() const
  {
    return comm_;
  }

  template<typename T,typename C>
  template<bool ignorePublic,typename RemoteIndexList,typename RemoteIndexMap>
  inline void ThreadParadigm<T,C>::buildRemote(const T* source, const T* target, RemoteIndexMap& remoteIndices, std::set<int>& neighbourIds,
                                               bool includeSelf)
  {
    // is the source different form the target?
    bool differentTarget(source != target);

    if(comm_.size()==1 && !(differentTarget || includeSelf))
      // nothing to do
      return;

    // create buffer to communicate indices
    typedef std::pair<const T*,const T*> IndicesPairType;
    comm_.template createBuffer<IndicesPairType>(tid_);
    comm_.template setBuffer<IndicesPairType>(IndicesPairType(source,target), tid_);

    // indices for which we receive
    RemoteIndexList* receive= new RemoteIndexList();
    // indices for which we send
    RemoteIndexList* send=0;

    typedef typename T::const_iterator const_iterator;
    typedef typename RemoteIndexList::MemberType RemoteIndex;

    if(differentTarget)
    {
      send = new RemoteIndexList();
      if(!(neighbourIds.empty())){
        for(size_t i = 0; i != comm_.size(); ++i)
        {
          const_iterator itEnd = source->end();
          for(const_iterator it = source->begin(); it != itEnd; ++it)
          {
            const_iterator itREnd = (comm_.template getBuffer<IndicesPairType>())[i].second->end();
            for(const_iterator itR = (comm_.template getBuffer<IndicesPairType>())[i].second->begin(); itR != itREnd; ++itR)
            {
              if(it->global() == itR->global())
              {
                neighbourIds.insert(i);
                PairType* pairTypePtr = new PairType(it->global(),LocalIndex(it->local().local(),it->local().attribute()));
                const RemoteIndex* remoteIdxPtr = new RemoteIndex(itR->local().attribute(), pairTypePtr);
                send->push_back(*remoteIdxPtr);
              }
            }
          }
        }
      }
      else
      {

      }

    }
    else
    {

    }

    comm_.template deleteBuffer<IndicesPairType>(tid_);

/*
    typedef typename T::const_iterator const_iterator;

    int noPublicSource=0;
    const const_iterator sourceItEnd=source->end();
    for(const_iterator it=source->begin(); it!=sourceItEnd; ++it)
      if(it->local().isPublic())
        ++noPublicSource;

    sourcePublish = (ignorePublic) ? source->size() : noPublicSource;

    if(sendTwo)
    {
      int noPublicTarget=0;
      const const_iterator targetItEnd=target->end();
        for(const_iterator it=target->begin(); it!=targetItEnd; ++it)
          if(it->local().isPublic())
            ++noPublicTarget;

      destPublish = (ignorePublic) ? target->size() : noPublicTarget;
    }
    else
      // we only need to send one set of indices
      destPublish = 0;

    int maxPublish, publish=sourcePublish+destPublish;

    // calculate maximum number of indices send
    comm_.template createBuffer<int>(tid_);
    comm_.template setBuffer<int>(publish, tid_);
    maxPublish = *(std::max_element(comm_.template getBuffer<int>().begin(), comm_.template getBuffer<int>().end()));
    comm_.template deleteBuffer<int>(tid_);

    //MPI_Allreduce(&publish, &maxPublish, 1, MPI_INT, MPI_MAX, comm_);

    // allocate buffers
    //typedef IndexPair<GlobalIndex,LocalIndex> PairType;

    PairType** destPairs;
    PairType** sourcePairs = new PairType*[sourcePublish>0 ? sourcePublish : 1];

    if(sendTwo)
      destPairs = new PairType*[destPublish>0 ? destPublish : 1];
    else
      destPairs=sourcePairs;

    char** buffer = new char*[2];
    int bufferSize;
    int position=0;
    int intSize;
    int charSize;

*/
//    unpackCreateRemote<RemoteIndexList,RemoteIndexMap>(buffer[0], sourcePairs, destPairs, remoteIndices, rank, sourcePublish, destPublish,
//                                                         bufferSize, sendTwo, includeSelf);

//template<typename RemoteIndexList,typename RemoteIndexMap>
//  inline void MPIParadigm<T>::unpackCreateRemote(char* p_in, PairType** sourcePairs, PairType** destPairs, RemoteIndexMap& remoteIndices,
//                                                 int remoteProc, int sourcePublish, int destPublish, int bufferSize, bool sendTwo,
//                                                 bool fromOurSelf)
//  {
    // unpack the number of indices we received
//    int noRemoteSource=-1, noRemoteDest=-1;
//    char twoIndexSets=0;
//    int position=0;
    // did we receive two index sets?
//    MPI_Unpack(p_in, bufferSize, &position, &twoIndexSets, 1, MPI_CHAR, comm_);
    // the number of source indices received
//    MPI_Unpack(p_in, bufferSize, &position, &noRemoteSource, 1, MPI_INT, comm_);
    // the number of destination indices received
//    MPI_Unpack(p_in, bufferSize, &position, &noRemoteDest, 1, MPI_INT, comm_);

    //MPI_Datatype type= MPITraits<PairType>::getType();
/*
    if(!twoIndexSets) {
      if(sendTwo) {
        send = new RemoteIndexList();
        // create both remote index sets simultaneously
        unpackIndices<RemoteIndexList>(*send, *receive, noRemoteSource, sourcePairs, sourcePublish, destPairs, destPublish, p_in, type,
                                       &position, bufferSize);
      }else{
        // we only need one list
        unpackIndices<RemoteIndexList>(*receive, noRemoteSource, sourcePairs, sourcePublish, p_in, type, &position, bufferSize, fromOurSelf);
        send=receive;
      }
    }else{

      int oldPos=position;
      // two index sets received
      unpackIndices<RemoteIndexList>(*receive, noRemoteSource, destPairs, destPublish, p_in, type, &position, bufferSize, fromOurSelf);
      if(!sendTwo)
        // unpack source entries again as destination entries
        position=oldPos;

      send = new RemoteIndexList();
      unpackIndices<RemoteIndexList>(*send, noRemoteDest, sourcePairs, sourcePublish, p_in, type, &position, bufferSize, fromOurSelf);
    }

    if(receive->empty() && send->empty()) {
      if(send==receive) {
        delete send;
      }else{
        delete send;
        delete receive;
      }
    }else{
      remoteIndices.insert(std::make_pair(remoteProc, std::make_pair(send,receive)));
    }

//  }
*/
/*
    // calculate buffer size
    MPI_Datatype type = MPITraits<PairType>::getType();

    MPI_Pack_size(maxPublish, type, comm_, &bufferSize);
    MPI_Pack_size(1, MPI_INT, comm_, &intSize);
    MPI_Pack_size(1, MPI_CHAR, comm_, &charSize);
    // our message will contain the following:
    // a bool wether two index sets where sent,
    // the size of the source and the dest indexset,
    // then the source and destination indices
    bufferSize += 2 * intSize + charSize;

    if(bufferSize<=0) bufferSize=1;

    buffer[0] = new char[bufferSize];
    buffer[1] = new char[bufferSize];

    // pack entries into buffer[0], p_out below
    MPI_Pack(&sendTwo, 1, MPI_CHAR, buffer[0], bufferSize, &position, comm_);

    // the number of indices we send for each index set
    MPI_Pack(&sourcePublish, 1, MPI_INT, buffer[0], bufferSize, &position, comm_);
    MPI_Pack(&destPublish, 1, MPI_INT, buffer[0], bufferSize, &position, comm_);

    // now pack the source indices and setup the destination pairs
    packEntries<ignorePublic>(sourcePairs, *source, buffer[0], type, bufferSize, &position, sourcePublish);
    // if necessary send the dest indices and setup the source pairs
    if(sendTwo)
      packEntries<ignorePublic>(destPairs, *target, buffer[0], type, bufferSize, &position, destPublish);

    // update remote indices for ourself
    if(sendTwo|| includeSelf)
      unpackCreateRemote<RemoteIndexList,RemoteIndexMap>(buffer[0], sourcePairs, destPairs, remoteIndices, rank, sourcePublish, destPublish,
                                                         bufferSize, sendTwo, includeSelf);

    neighbourIds.erase(rank);

    if(neighbourIds.size()==0)
    {
      dvverb<<rank<<": Sending messages in a ring"<<std::endl;
      // send messages in ring
      for(int proc=1; proc<procs; proc++) {
        // pointers to the current input and output buffers
        char* p_out = buffer[1-(proc%2)];
        char* p_in = buffer[proc%2];

        MPI_Status status;
        if(rank%2==0) {
          MPI_Ssend(p_out, bufferSize, MPI_PACKED, (rank+1)%procs, commTag_, comm_);
          MPI_Recv(p_in, bufferSize, MPI_PACKED, (rank+procs-1)%procs, commTag_, comm_, &status);
        }else{
          MPI_Recv(p_in, bufferSize, MPI_PACKED, (rank+procs-1)%procs, commTag_, comm_, &status);
          MPI_Ssend(p_out, bufferSize, MPI_PACKED, (rank+1)%procs, commTag_, comm_);
        }

        // the process these indices are from
        int remoteProc = (rank+procs-proc)%procs;

        unpackCreateRemote<RemoteIndexList,RemoteIndexMap>(p_in, sourcePairs, destPairs, remoteIndices, remoteProc, sourcePublish,
                                                           destPublish, bufferSize, sendTwo);

      }

    }
    else
    {
      MPI_Request* requests=new MPI_Request[neighbourIds.size()];
      MPI_Request* req=requests;

      typedef typename std::set<int>::size_type size_type;
      size_type noNeighbours=neighbourIds.size();

      // setup sends
      for(std::set<int>::iterator neighbour=neighbourIds.begin();
          neighbour!= neighbourIds.end(); ++neighbour) {
        // only send the information to the neighbouring processors
        MPI_Issend(buffer[0], position , MPI_PACKED, *neighbour, commTag_, comm_, req++);
      }

      // test for received messages
      for(size_type received=0; received <noNeighbours; ++received)
      {
        MPI_Status status;
        // probe for next message
        MPI_Probe(MPI_ANY_SOURCE, commTag_, comm_, &status);
        int remoteProc=status.MPI_SOURCE;
        int size;
        MPI_Get_count(&status, MPI_PACKED, &size);
        // receive message
        MPI_Recv(buffer[1], size, MPI_PACKED, remoteProc, commTag_, comm_, &status);

        unpackCreateRemote<RemoteIndexList,RemoteIndexMap>(buffer[1], sourcePairs, destPairs, remoteIndices, remoteProc, sourcePublish,
                                                           destPublish, bufferSize, sendTwo);
      }
      // wait for completion of pending requests
      MPI_Status* statuses = new MPI_Status[neighbourIds.size()];

      if(MPI_ERR_IN_STATUS==MPI_Waitall(neighbourIds.size(), requests, statuses)) {
        for(size_type i=0; i < neighbourIds.size(); ++i)
          if(statuses[i].MPI_ERROR!=MPI_SUCCESS) {
            std::cerr<<rank<<": MPI_Error occurred while receiving message."<<std::endl;
            MPI_Abort(comm_, 999);
          }
      }
      delete[] requests;
      delete[] statuses;
    }


    // delete allocated memory
    if(destPairs!=sourcePairs)
      delete[] destPairs;

    delete[] sourcePairs;
    delete[] buffer[0];
    delete[] buffer[1];
    delete[] buffer;
*/
  }

  /** @} */

}

#endif
