// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADPARALLELPARADIGM_HH
#define DUNE_THREADPARALLELPARADIGM_HH

#include "indexset.hh"
#include "plocalindex.hh"
#include <dune/common/stdstreams.hh>
#include <map>
#include <set>
#include <iostream>
#include <thread>
#include <mutex>

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
  class ThreadCommunicator
  {
    public:
      /** @brief Constructor. */
      inline ThreadCommunicator(const size_t& size);

      /** @brief Default constructor. */
      ThreadCommunicator()
      {}

      /** @brief Set the number of threads. */
      inline void setSize(const size_t& size);

      /** @brief Destructor. */
      ~ThreadCommunicator()
      {}

      /** @brief Get the number of threads. */
      inline size_t size() const;

    private:
      /** @brief The number of threads. */
      size_t size_;

      /** @brief Buffer to exchange int among threads. */
      std::vector<int> bufferint_;

  };

  /**
   * @brief ThreadParadigm.
   * @tparam T The type of the underlying index set.
   */
  template<class T>
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
    typedef ThreadCommunicator CommType;

    /** @brief Constructor. */
    inline ThreadParadigm(const size_t& tid, const size_t& size);

    /** @brief Default constructor. */
    ThreadParadigm()
    {}

     /** @brief Set the paradigm we work with. */
    inline void setParadigm(const size_t& tid, const size_t& size);

    /** @brief Destructor. */
    ~ThreadParadigm()
    {}

    /** @brief Get the thread ID. */
    inline size_t threadID() const;

    /** @brief Get the total number of threads. */
    inline size_t numThreads() const;

    /** @brief Get the communicator. */
    inline CommType communicator() const;

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

    /** @brief The thread ID. */
    size_t tid_;

    /** @brief The number of threads. */
    size_t size_;

    /** @brief The index pair type. */
    typedef IndexPair<GlobalIndex,LocalIndex> PairType;

    /** @brief Buffer to exchange int among threads. */
    static std::vector<int> bufferint_;
  };

  /** @} */

  template<typename T>
  inline ThreadParadigm<T>::ThreadParadigm(const size_t& tid, const size_t& size) : tid_(tid), size_(size)
  {}

  template<typename T>
  inline void ThreadParadigm<T>::setParadigm(const size_t& tid, const size_t& size)
  {
    tid_ = tid;
    size_ = size;
  }

  template<typename T>
  inline size_t ThreadParadigm<T>::threadID() const
  {
    return tid_;
  }

  template<typename T>
  inline size_t ThreadParadigm<T>::numThreads() const
  {
    return size_;
  }

  template<typename T>
  template<bool ignorePublic,typename RemoteIndexList,typename RemoteIndexMap>
  inline void ThreadParadigm<T>::buildRemote(const T* source, const T* target, RemoteIndexMap& remoteIndices, std::set<int>& neighbourIds,
                                             bool includeSelf)
  {
    // number of local indices to publish
    // the indices of the destination will be send
    int sourcePublish, destPublish;

    // do we need to send two index sets?
    char sendTwo = (source != target);

    if(size_==1 && !(sendTwo || includeSelf))
      // nothing to communicate
      return;

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
/*
    // calculate maximum number of indices send
    MPI_Allreduce(&publish, &maxPublish, 1, MPI_INT, MPI_MAX, comm_);

    // allocate buffers
    typedef IndexPair<GlobalIndex,LocalIndex> PairType;

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
