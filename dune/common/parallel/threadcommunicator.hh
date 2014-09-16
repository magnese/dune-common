
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADCOMMUNICATOR
#define DUNE_THREADCOMMUNICATOR

#include "interface.hh"
#include "communicator.hh"
#include <vector>
#include <set>
#include <algorithm>
#include <utility>

namespace Dune
{
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Provides the implementation for syncing distributed data via thread communication.
   * @author Marco Agnese, Markus Blatt
   */

  /** @brief Thread communicator which uses a coloring algorithm to send and receive data. */
  class ThreadCommunicator
  {
  public:
    /** @brief Constructor. */
    ThreadCommunicator()
    {}

    /** @brief Build the buffers and the information for the communication process. */
    template<class Data, class I>
    void build(const I& interface);

    /**
      * @brief Build the buffers and information for the communication process.
      * @param source The source in a forward send. The values will be copied from here to the send buffers.
      * @param target The target in a forward send. The received values will be copied to here.
      */
    template<class Data, class I>
    void build(const Data& source, const Data& target, const I& interface);

    /** @brief Empty method needed to satisfy the communicator interface. */
    void free()
    {}

    /** @brief Send and receive Data. */
    template<class GatherScatter, bool FORWARD, class Data>
    void sendRecv(const Data& source, Data& target);

    /** @brief Destructor. */
    ~ThreadCommunicator()
    {}

  private:
    /** @brief The type of the map that maps interface information to processors. */
    typedef std::map<int,std::pair<InterfaceInformation,InterfaceInformation> > InterfaceMap;

    /** @brief The pointer to the collective communication. */
    ThreadCollectiveCommunication* collcommptr_;

    size_t threadID_;

    /** @brief The interface we currently work with. */
    InterfaceMap interfaces_;

    /** @brief The colors of the threads. colors_[i] = color of thread i. */
    std::vector<int> colors_;

    /** @brief Number of different colors present in colors_. */
    unsigned int numcolors_;

    /** @brief Compute the coloring scheme.*/
    template<class I>
    void computeColoring(const I& interface);

    /** @brief  Functors to update the values in the target. */
    template<class Data, class GatherScatter,typename IndexedTypeFlag>
    struct Updater
    {};

    /** @brief Functor specialization for SizeOne. */
    template<class Data, class GatherScatter>
    struct Updater<Data,GatherScatter,SizeOne>
    {
      inline void operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target);
    };

    /** @brief Functor specialization for VariableSize. */
    template<class Data, class GatherScatter>
    struct Updater<Data,GatherScatter,VariableSize>
    {
      inline void operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target);
    };

  };

  /** @} */

  template<typename Data, typename I>
  void ThreadCommunicator::build(const I& interface)
  {
    collcommptr_ = &(interface.remoteIndices().parallelParadigm().collCommunicator());
    threadID_ = interface.remoteIndices().parallelParadigm().threadID();
    interfaces_ = interface.interfaces();
    computeColoring(interface);
  }

  template<typename Data, typename I>
  void ThreadCommunicator::build(const Data& source, const Data& target, const I& interface)
  {
    collcommptr_ = &(interface.remoteIndices().parallelParadigm().collCommunicator());
    threadID_ = interface.remoteIndices().parallelParadigm().threadID();
    interfaces_ = interface.interfaces();
    computeColoring(interface);
  }

  template<typename I>
  void ThreadCommunicator::computeColoring(const I& interface)
  {
    typedef typename I::RemoteIndicesType RemoteIndicesType;
    typedef typename RemoteIndicesType::ParallelParadigm ParallelParadigm;
    typedef typename ParallelParadigm::CollectiveCommunicationType CollectiveCommunicationType;

    RemoteIndicesType& remoteIndices = interface.remoteIndices();
    ParallelParadigm& parallelParadigm = remoteIndices.parallelParadigm();
    CollectiveCommunicationType& collComm = parallelParadigm.collCommunicator();

    const size_t numThreads = parallelParadigm.numThreads();
    const size_t tid = parallelParadigm.threadID();

    colors_.clear();
    colors_.resize(numThreads, -1);
    colors_[0] = 0;

    if(numThreads > 1)
    {
      // compute adiajency matrix of the graph rappresenting the interaction between threads
      std::vector<std::vector<int>> adjMatrix(numThreads,std::vector<int>(numThreads,-1));
      // create buffer to communicate neighbours
      const std::set<int>* neighboursPtr = &(remoteIndices.getNeighbours());
      collComm.template createBuffer<const std::set<int>*>();
      collComm.template setBuffer<const std::set<int>*>(neighboursPtr, tid);

      typedef typename std::set<int>::iterator SetIterType;
      for(size_t i = 0; i != numThreads; ++i)
      {
        const std::set<int>* ptr = (collComm.template getBuffer<const std::set<int>*>())[i];
        SetIterType itEnd = ptr->end();
        for(SetIterType it = ptr->begin(); it != itEnd; ++it)
          adjMatrix[i][*it] = 1;
      }
      collComm.template deleteBuffer<const std::set<int>*>();

      // compute colouring with a greedy algorithm
      for(size_t i = 1; i != numThreads; ++i)
      {
        int color = 0;
        for(size_t j = 0; j != numThreads; ++j)
        {
          if(adjMatrix[i][j] == 1)
          {
            if(colors_[j] > -1)
            {
              color = std::max(color,colors_[j]+1);
            }
          }
        }
        colors_[i] = color;
      }

    }

    numcolors_ = *(std::max_element(colors_.begin(),colors_.end()))+1;
  }

  template<typename Data, typename GatherScatter>
  inline void ThreadCommunicator::Updater<Data,GatherScatter,SizeOne>::operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target)
  {
    GatherScatter::scatter(target,GatherScatter::gather(source,idxSource),idxTarget);
  }

  template<typename Data, typename GatherScatter>
  inline void ThreadCommunicator::Updater<Data,GatherScatter,VariableSize>::operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target)
  {
    for(size_t j = 0; j != CommPolicy<Data>::getSize(source,idxSource); ++j)
      GatherScatter::scatter(target,GatherScatter::gather(source,idxSource,j),idxTarget,j);
  }

  template<typename GatherScatter, bool FORWARD, typename Data>
  void ThreadCommunicator::sendRecv(const Data& source, Data& target)
  {
    typedef typename CommPolicy<Data>::IndexedTypeFlag Flag;
    Updater<Data,GatherScatter,Flag> updater;

    typedef typename InterfaceMap::const_iterator const_iterator;

    // create the buffer to communicate data
    typedef std::pair<Data*,const InterfaceMap*> BufferType;
    collcommptr_->template createBuffer<BufferType>();
    collcommptr_->template setBuffer<BufferType>(BufferType(&target,&interfaces_), threadID_);

    if(FORWARD)
    {
      for(int color = 0; color != numcolors_ ; ++color)
      {
        if(colors_[threadID_] == color)
        {
          const_iterator itEnd = interfaces_.end();
          for(const_iterator it = interfaces_.begin(); it != itEnd; ++it)
          {
            size_t size = it->second.first.size();
            Data& dest = *(((collcommptr_->template getBuffer<BufferType>())[it->first]).first);
            const_iterator itDest =  (((collcommptr_->template getBuffer<BufferType>())[it->first]).second)->find(threadID_);
            for(size_t i=0; i < size; i++)
              updater(it->second.first[i],source,itDest->second.second[i],dest);
          }
        }
        collcommptr_->barrier();
      }
    }
    else
    {
      for(int color = 0; color != numcolors_ ; ++color)
      {
        if(colors_[threadID_] == color)
        {
          const_iterator itEnd = interfaces_.end();
          for(const_iterator it = interfaces_.begin(); it != itEnd; ++it)
          {
            size_t size = it->second.second.size();
            Data& dest = *(((collcommptr_->template getBuffer<BufferType>())[it->first]).first);
            const_iterator itDest = (((collcommptr_->template getBuffer<BufferType>())[it->first]).second)->find(threadID_);
            for(size_t i=0; i < size; i++)
              updater(it->second.second[i],source,itDest->second.first[i],dest);
          }
        }
        collcommptr_->barrier();
      }
    }

    collcommptr_->template deleteBuffer<BufferType>();
  }


}

#endif
