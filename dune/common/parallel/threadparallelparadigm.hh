// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADPARALLELPARADIGM_HH
#define DUNE_THREADPARALLELPARADIGM_HH

#include "indexset.hh"
#include "plocalindex.hh"
#include <dune/common/stdstreams.hh>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>
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

  /** @brief The communicator for thread. */
  class THREAD_Comm
  {
  };

  /** @brief ThreadCommunicator allows communication between threads using shared memeory. */
  class ThreadCollectiveCommunication
  {
    public:
      /** @brief The type of the communicator. */
      typedef THREAD_Comm CommType;

      /** @brief Default constructor. */
      ThreadCollectiveCommunication(const size_t& numThreads);

      /** @brief Destructor. */
      inline ~ThreadCollectiveCommunication()
      {}

      /** @brief Get the communicator. */
      inline CommType communicator();

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
      inline std::vector<T>& getBuffer() const;

      /** @brief Delete buffer. */
      template<class T>
      inline void deleteBuffer();

    private:
      /** @brief Thread communicator. */
      CommType comm_;

      /** @brief Number of threads. */
      size_t size_;

      /** @brief Flag used by std::call_once function. */
      std::once_flag bufferflag_;

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

  /** @brief ThreadParadigm. */
  class ThreadParadigm
  {

  public:
    /** @brief The type of the collective communication. */
    typedef ThreadCollectiveCommunication CollectiveCommunicationType;

    /** @brief The type of the communicator. */
    typedef typename CollectiveCommunicationType::CommType CommType;

    /** @brief Constructor. */
    inline ThreadParadigm(CollectiveCommunicationType& collComm, const size_t& tid);

    /** @brief Null communicator. */
    constexpr static CommType nullComm = CommType();

    /** @brief Default constructor. */
    ThreadParadigm()
    {}

     /** @brief Set the paradigm we work with. */
    inline void setParadigm(CollectiveCommunicationType& collComm, const size_t& tid);

    /** @brief Destructor. */
    ~ThreadParadigm()
    {}

    /** @brief Get the thread ID. */
    inline size_t threadID() const;

    /** @brief Get the number of threads. */
    inline size_t numThreads() const;

    /** @brief Get the collective communicator. */
    inline CollectiveCommunicationType& collCommunicator() const;

    /** @brief Get the communicator. */
    inline CommType communicator();

    //! \todo Please finsih to doc me.
    /**
     * @brief Build the remote mapping. If the template parameter ignorePublic is true all indices will be treated as public.
     * @param includeSelf If true, sending from indices of the processor to other indices on the same processor is enabled even
     * if the same indexset is used on both the sending and receiving side.
     */
    template<bool ignorePublic,class ParallelIndexSet, class RemoteIndexList, class RemoteIndexMap = std::map<int, std::pair<RemoteIndexList*,RemoteIndexList*> > >
    inline void buildRemote(const ParallelIndexSet* source, const ParallelIndexSet* target, RemoteIndexMap& remoteIndices,
                            std::set<int>& neighbourIds, bool includeSelf);

  private:
    /** copying is forbidden. */
    ThreadParadigm(const ThreadParadigm&)
    {}

    /** @brief The collective communication. */
    CollectiveCommunicationType& colcomm_;

    /** @brief The thread ID. */
    size_t tid_;

    /** @brief Given a source and a target, it creates the corresponding RemoteIndexList. */
    template<bool ignorePublic,class ParallelIndexSet,class RemoteIndexList>
    inline RemoteIndexList* createRemoteIndexList(const ParallelIndexSet* source,const ParallelIndexSet* target);
  };

  /** @} */

  ThreadCollectiveCommunication::ThreadCollectiveCommunication(const size_t& numThreads) :
    comm_(), size_(numThreads), bufferptr_(nullptr), mtx_(), count_(0), seccount_(0), condvar_()
  {}

  inline const size_t& ThreadCollectiveCommunication::size() const
  {
    return size_;
  }

  inline typename ThreadCollectiveCommunication::CommType ThreadCollectiveCommunication::communicator()
  {
    return comm_;
  }

  inline void ThreadCollectiveCommunication::barrier()
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

  template<typename T>
  inline void ThreadCollectiveCommunication::createBuffer_()
  {
    bufferptr_ = new std::vector<T>(size_);
  }

  template<typename T>
  inline void ThreadCollectiveCommunication::createBuffer()
  {
    std::call_once(bufferflag_,&ThreadCollectiveCommunication::createBuffer_<T>,this);
  }

  template<typename T>
  inline void ThreadCollectiveCommunication::setBuffer(const T& value, const size_t& tid)
  {
    barrier(); // checkpoint: the buffer is allocated
    (*(static_cast<std::vector<T>*>(bufferptr_)))[tid] = value;
    barrier(); // checkpoitn: the buffer is set
  }

  template<typename T>
  inline std::vector<T>& ThreadCollectiveCommunication::getBuffer() const
  {
    return *(static_cast<std::vector<T>*>(bufferptr_));
  }

  template<typename T>
  inline void ThreadCollectiveCommunication::deleteBuffer_()
  {
    delete static_cast<std::vector<T>*>(bufferptr_);
    bufferptr_ = nullptr;
  }

  template<typename T>
  inline void ThreadCollectiveCommunication::deleteBuffer()
  {
    barrier(); // checkpoint: the buffer isn't needed anymore
    std::call_once(bufferflag_,&ThreadCollectiveCommunication::deleteBuffer_<T>,this);
    barrier(); // checkpoint: the buffer is free
  }

  inline ThreadParadigm::ThreadParadigm(CollectiveCommunicationType& colComm, const size_t& tid) : colcomm_(colComm), tid_(tid)
  {}

  inline void ThreadParadigm::setParadigm(CollectiveCommunicationType& colComm, const size_t& tid)
  {
    colcomm_ = colComm;
    tid_ = tid;
  }

  inline size_t ThreadParadigm::threadID() const
  {
    return tid_;
  }

  inline size_t ThreadParadigm::numThreads() const
  {
    return colcomm_.size();
  }

  inline typename ThreadParadigm::CollectiveCommunicationType& ThreadParadigm::collCommunicator() const
  {
    return colcomm_;
  }

  inline typename ThreadParadigm::CommType ThreadParadigm::communicator()
  {
    return colcomm_.communicator();
  }

  template<bool ignorePublic,typename ParallelIndexSet,typename RemoteIndexList>
  inline RemoteIndexList* ThreadParadigm::createRemoteIndexList(const ParallelIndexSet* source,const ParallelIndexSet* target)
  {
    // index list
    RemoteIndexList* remoteIndexList = new RemoteIndexList();

    typedef typename ParallelIndexSet::const_iterator const_iterator;
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

  template<bool ignorePublic,typename ParallelIndexSet,typename RemoteIndexList,typename RemoteIndexMap>
  inline void ThreadParadigm::buildRemote(const ParallelIndexSet* source, const ParallelIndexSet* target, RemoteIndexMap& remoteIndices, std::set<int>& neighbourIds, bool includeSelf)
  {
    // is the source different form the target?
    bool differentTarget(source != target);

    if(colcomm_.size()==1 && !(differentTarget || includeSelf))
      // nothing to do
      return;

    // create buffer to communicate indices
    typedef std::pair<const ParallelIndexSet*,const ParallelIndexSet*> IndicesPairType;
    colcomm_.template createBuffer<IndicesPairType>();
    colcomm_.template setBuffer<IndicesPairType>(IndicesPairType(source,target), tid_);

    // indices list for sending and receive
    RemoteIndexList* send(nullptr);
    RemoteIndexList* receive(nullptr);

    if(neighbourIds.empty())
    {
      for(size_t remoteProc = 0; remoteProc != colcomm_.size(); ++remoteProc)
      {
        if(includeSelf || ((!includeSelf)&&(remoteProc!=tid_)))
        {
          send = createRemoteIndexList<ignorePublic,ParallelIndexSet,RemoteIndexList>(source,(colcomm_.template getBuffer<IndicesPairType>())[remoteProc].second);

          if(!(send->empty()))
            neighbourIds.insert(remoteProc);
          if(differentTarget && (!(send->empty())))
            receive  = createRemoteIndexList<ignorePublic,ParallelIndexSet,RemoteIndexList>(target,(colcomm_.template getBuffer<IndicesPairType>())[remoteProc].first);
          else
            receive=send;

          remoteIndices.insert(std::make_pair(remoteProc, std::make_pair(send,receive)));
        }
      }
    }
    else
    {
      for(size_t remoteProc = 0; remoteProc != colcomm_.size(); ++remoteProc)
      {
        if(includeSelf || ((!includeSelf)&&(remoteProc!=tid_)))
        {
          if(neighbourIds.find(remoteProc)!=neighbourIds.end())
          {
            send = createRemoteIndexList<ignorePublic,ParallelIndexSet,RemoteIndexList>(source,(colcomm_.template getBuffer<IndicesPairType>())[remoteProc].second);
            if(differentTarget)
              receive  = createRemoteIndexList<ignorePublic,ParallelIndexSet,RemoteIndexList>(target,(colcomm_.template getBuffer<IndicesPairType>())[remoteProc].first);
            else
              receive=send;

            remoteIndices.insert(std::make_pair(remoteProc, std::make_pair(send,receive)));
          }
        }
      }

    }

    colcomm_.template deleteBuffer<IndicesPairType>();

  }

  /** @} */

}

#endif
