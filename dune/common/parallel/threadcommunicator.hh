
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADCOMMUNICATOR
#define DUNE_THREADCOMMUNICATOR

#include <vector>
#include <set>
#include <algorithm>
#include <utility>
#include "communicator.hh"

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

  /** @brief Thread communicator whihc uses a coloring algorithm to send and receive data. */
  template<class I>
  class ThreadCommunicator
  {
  public:
    /** @brief The type of the interface. */
    typedef I InterfaceType;

    /**
     * @brief Constructor.
     * @param interface The interface that defines what indices are to be communicated.
     */
    ThreadCommunicator(InterfaceType& interface);

    /** @brief Destructor. */
    ~ThreadCommunicator()
    {}

    /** @brief Empty method needed to satisfy the communicator interface. */
    void build()
    {}

    /** @brief Empty method needed to satisfy the communicator interface. */
    template<class Data>
    void build(const Data& source, const Data& target)
    {}

    /** @brief Empty method needed to satisfy the communicator interface. */
    void free()
    {}

    /** @brief Send and receive Data. */
    template<class GatherScatter, bool FORWARD, class Data>
    void sendRecv(const Data& source, Data& target);

  private:
    /** @brief The type of remote indices. */
    typedef typename InterfaceType::RemoteIndicesType RemoteIndicesType;

    /** @brief The type of the parallel paradigm we use. */
    typedef typename RemoteIndicesType::ParallelParadigm ParallelParadigm;

    /** @brief The type of the collective communication. */
    typedef typename ParallelParadigm::CollectiveCommunicationType CollectiveCommunicationType;

    /** @brief The type of the map that maps interface information to processors. */
    typedef typename InterfaceType::InformationMap InterfaceMap;

    /** @brief The interface we currently work with. */
    InterfaceType& interface_;

    /** @brief The interface map. */
    InterfaceMap& interfaces_;

    /** @brief The colors of the threads. colors_[i] = color of thread i. */
    std::vector<int> colors_;

    /** @brief Number of different colors present in colors_. */
    unsigned int numcolors_;

    /** @brief Compute the coloring scheme.*/
    void computeColoring();

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

  template<typename I>
  inline ThreadCommunicator<I>::ThreadCommunicator(InterfaceType& interface) : interface_(interface), interfaces_(interface_.interfaces())
  {
    computeColoring();
  }

  template<typename I>
  void ThreadCommunicator<I>::computeColoring()
  {
    RemoteIndicesType& remoteIndices = interface_.remoteIndices();
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

  template<typename I>
  template<typename Data, typename GatherScatter>
  inline void ThreadCommunicator<I>::Updater<Data,GatherScatter,SizeOne>::operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target)
  {
    GatherScatter::scatter(target,GatherScatter::gather(source,idxSource),idxTarget);
  }

  template<typename I>
  template<typename Data, typename GatherScatter>
  inline void ThreadCommunicator<I>::Updater<Data,GatherScatter,VariableSize>::operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target)
  {
    for(size_t j = 0; j != CommPolicy<Data>::getSize(source,idxSource); ++j)
      GatherScatter::scatter(target,GatherScatter::gather(source,idxSource,j),idxTarget,j);
  }

  template<typename I>
  template<typename GatherScatter, bool FORWARD, typename Data>
  void ThreadCommunicator<I>::sendRecv(const Data& source, Data& target)
  {
    RemoteIndicesType& remoteIndices = interface_.remoteIndices();
    ParallelParadigm& parallelParadigm = remoteIndices.parallelParadigm();
    CollectiveCommunicationType& collComm = parallelParadigm.collCommunicator();

    typedef typename CommPolicy<Data>::IndexedTypeFlag Flag;
    Updater<Data,GatherScatter,Flag> updater;

    typedef typename InterfaceMap::const_iterator const_iterator;

    const size_t tid = parallelParadigm.threadID();

    // create the buffer to communicate data
    typedef std::pair<Data*,const InterfaceMap*> BufferType;
    collComm.template createBuffer<BufferType>();
    collComm.template setBuffer<BufferType>(BufferType(&target,&interfaces_), tid);

    if(FORWARD)
    {
      for(int color = 0; color != numcolors_ ; ++color)
      {
        if(colors_[tid] == color)
        {
          const_iterator itEnd = interfaces_.end();
          for(const_iterator it = interfaces_.begin(); it != itEnd; ++it)
          {
            size_t size = it->second.first.size();
            Data& dest = *(((collComm.template getBuffer<BufferType>())[it->first]).first);
            const_iterator itDest =  (((collComm.template getBuffer<BufferType>())[it->first]).second)->find(tid);
            for(size_t i=0; i < size; i++)
              updater(it->second.first[i],source,itDest->second.second[i],dest);
          }
        }
        collComm.barrier();
      }
    }
    else
    {
      for(int color = 0; color != numcolors_ ; ++color)
      {
        if(colors_[tid] == color)
        {
          const_iterator itEnd = interfaces_.end();
          for(const_iterator it = interfaces_.begin(); it != itEnd; ++it)
          {
            size_t size = it->second.second.size();
            Data& dest = *(((collComm.template getBuffer<BufferType>())[it->first]).first);
            const_iterator itDest =  (((collComm.template getBuffer<BufferType>())[it->first]).second)->find(tid);
            for(size_t i=0; i < size; i++)
              updater(it->second.second[i],source,itDest->second.first[i],dest);
          }
        }
        collComm.barrier();
      }
    }

    collComm.template deleteBuffer<BufferType>();
  }


}

#endif
