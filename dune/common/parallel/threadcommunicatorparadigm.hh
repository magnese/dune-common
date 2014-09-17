
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
  class ThreadCommunicatorParadigm
  {
  public:
    /** @brief Constructor. */
    ThreadCommunicatorParadigm()
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
    ~ThreadCommunicatorParadigm()
    {}

  private:
    /** @brief The type of the map that maps interface information to processors. */
    typedef std::map<int,std::pair<InterfaceInformation,InterfaceInformation> > InterfaceMap;

    /** @brief The pointer to the collective communication. */
    ThreadCollectiveCommunication* collcommptr_;

    size_t threadID_;

    /** @brief The colors of the threads. colors_[i] = color of thread i. */
    std::vector<int>* colorsptr_;

    /** @brief Number of different colors present in colors_. */
    unsigned int numcolors_;

    /** @brief The interface we currently work with. */
    InterfaceMap interfaces_;

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
  void ThreadCommunicatorParadigm::build(const I& interface)
  {
    collcommptr_ = &(interface.parallelParadigm().collCommunicator());
    threadID_ = interface.parallelParadigm().threadID();
    colorsptr_ = &(interface.parallelParadigm().colors());
    numcolors_ = interface.parallelParadigm().numColors();
    interfaces_ = interface.interfaces();
  }

  template<typename Data, typename I>
  void ThreadCommunicatorParadigm::build(const Data& source, const Data& target, const I& interface)
  {
    build<Data,I>(interface);
  }

  template<typename Data, typename GatherScatter>
  inline void ThreadCommunicatorParadigm::Updater<Data,GatherScatter,SizeOne>::operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target)
  {
    GatherScatter::scatter(target,GatherScatter::gather(source,idxSource),idxTarget);
  }

  template<typename Data, typename GatherScatter>
  inline void ThreadCommunicatorParadigm::Updater<Data,GatherScatter,VariableSize>::operator()(const size_t& idxSource, const Data& source, const size_t& idxTarget, Data& target)
  {
    for(size_t j = 0; j != CommPolicy<Data>::getSize(source,idxSource); ++j)
      GatherScatter::scatter(target,GatherScatter::gather(source,idxSource,j),idxTarget,j);
  }

  template<typename GatherScatter, bool FORWARD, typename Data>
  void ThreadCommunicatorParadigm::sendRecv(const Data& source, Data& target)
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
        if((*colorsptr_)[threadID_] == color)
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
        if((*colorsptr_)[threadID_] == color)
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
