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
      inline void createBuffer();

      /** @brief Set a value into the buffer. */
      template<class T>
      inline void setBuffer(const T& value, const size_t& tid);

      /** @brief Get buffer. */
      template<class T>
      inline std::array<T,numThreads>& getBuffer() const;

      /** @brief Delete buffer. */
      template<class T>
      inline void deleteBuffer();

    private:
      /** @brief Number of threads. */
      const size_t size_;

      /** @brief Flag used by std::call_once function. */
      std::once_flag bufferflag_;

      /** @brief Status of the buffer. */
      bool bufferready_;

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

      /** @brief Allocate the buffer. */
      template<class T>
      inline void createBuffer_();

      /** @brief Deallocate the buffer. */
      template<class T>
      inline void deleteBuffer_();
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

    /** @brief Given a source and a target, it creates the corresponding RemoteIndexList. */
    template<bool ignorePublic,typename RemoteIndexList>
    inline RemoteIndexList* createRemoteIndexList(const T* source,const T* target);

  };

  /** @} */

  template<size_t numThreads>
  inline ThreadCommunicator<numThreads>::ThreadCommunicator() : size_(numThreads), bufferready_(false), bufferptr_(nullptr), mtx_(), count_(0), seccount_(0), condvar_()
  {}

  template<size_t numThreads>
  inline const size_t& ThreadCommunicator<numThreads>::size() const
  {
    return size_;
  }

  //TODO: FIX ME! sometimes it leads to a segmentaiton fault (I think)
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
      count_ = 0; // reset counter for later use of the barrier
      // increasing the second counter allows waiting threads to exit
      ++seccount_;
      condvar_.notify_all();
    }
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::createBuffer_()
  {
    bufferptr_ = new std::array<T,numThreads>();
    bufferready_ = true;
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::createBuffer()
  {
    std::call_once(bufferflag_,&ThreadCommunicator<numThreads>::createBuffer_<T>,this);
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::setBuffer(const T& value, const size_t& tid)
  {
    while(!bufferready_);
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
  inline void ThreadCommunicator<numThreads>::deleteBuffer_()
  {
    bufferready_ = false;
    delete static_cast<std::array<T,numThreads>*>(bufferptr_);
    bufferptr_ = nullptr;
  }

  template<size_t numThreads>
  template<typename T>
  inline void ThreadCommunicator<numThreads>::deleteBuffer()
  {
    std::call_once(bufferflag_,&ThreadCommunicator<numThreads>::deleteBuffer_<T>,this);
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
  template<bool ignorePublic,typename RemoteIndexList>
  inline RemoteIndexList* ThreadParadigm<T,C>::createRemoteIndexList(const T* source,const T* target)
  {
    // index list
    RemoteIndexList* remoteIndexList = new RemoteIndexList();

    typedef typename T::const_iterator const_iterator;
    typedef typename RemoteIndexList::MemberType RemoteIndex;

    const_iterator itSEnd = source->end();
    for(const_iterator itS = source->begin(); itS != itSEnd; ++itS)
    {
      const_iterator itTEnd = target ->end();
      for(const_iterator itT = target ->begin(); itT != itTEnd; ++itT)
      {

        if(itS->global() == itT->global())
        {
          if(itS->local().isPublic() || ignorePublic)
          {
            const RemoteIndex* remoteIdxPtr = new RemoteIndex(itT->local().attribute(), &(*itS));
            remoteIndexList->push_back(*remoteIdxPtr);
          }
        }

      }
    }

    return remoteIndexList;
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
    comm_.template createBuffer<IndicesPairType>();
    comm_.template setBuffer<IndicesPairType>(IndicesPairType(source,target), tid_);

    // indices list for sending and receive
    RemoteIndexList* send(nullptr);
    RemoteIndexList* receive(nullptr);

    if(neighbourIds.empty())
    {
      for(size_t remoteProc = 0; remoteProc != comm_.size(); ++remoteProc)
      {
        if(includeSelf || ((!includeSelf)&&(remoteProc!=tid_)))
        {
          send = createRemoteIndexList<ignorePublic,RemoteIndexList>(source,(comm_.template getBuffer<IndicesPairType>())[remoteProc].second);

          if(!(send->empty()))
            neighbourIds.insert(remoteProc);
          if(differentTarget && (!(send->empty())))
            receive  = createRemoteIndexList<ignorePublic,RemoteIndexList>(target,(comm_.template getBuffer<IndicesPairType>())[remoteProc].first);
          else
            receive=send;

          remoteIndices.insert(std::make_pair(remoteProc, std::make_pair(send,receive)));
        }
      }
    }
    else
    {
      for(size_t remoteProc = 0; remoteProc != comm_.size(); ++remoteProc)
      {
        if(includeSelf || ((!includeSelf)&&(remoteProc!=tid_)))
        {
          if(neighbourIds.find(remoteProc)!=neighbourIds.end())
          {
            send = createRemoteIndexList<ignorePublic,RemoteIndexList>(source,(comm_.template getBuffer<IndicesPairType>())[remoteProc].second);
            if(differentTarget)
              receive  = createRemoteIndexList<ignorePublic,RemoteIndexList>(target,(comm_.template getBuffer<IndicesPairType>())[remoteProc].first);
            else
              receive=send;

            remoteIndices.insert(std::make_pair(remoteProc, std::make_pair(send,receive)));
          }
        }
      }

    }

    comm_.template deleteBuffer<IndicesPairType>();

  }

  /** @} */

}

#endif
