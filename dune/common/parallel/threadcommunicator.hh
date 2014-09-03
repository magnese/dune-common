
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADCOMMUNICATOR
#define DUNE_THREADCOMMUNICATOR

#include <vector>
#include <set>
#include <algorithm>
#include <utility>
// #include "remoteindices.hh"
// #include "interface.hh"
// #include <dune/common/exceptions.hh>
// #include <dune/common/typetraits.hh>
// #include <dune/common/stdstreams.hh>

namespace Dune
{
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Provides utility classes for syncing distributed data via thread communication.
   * @author Marco Agnese, Markus Blatt
   */

  /** @brief Communicator that uses a coloring algorithm to gather and scatter the data to be send or received. */
  template<class I>
  class ThreadCommunicator
  {
  public:
    /** @brief The type of the interface. */
    typedef I InterfaceType;

    /** @brief The type of remote indices. */
    typedef typename InterfaceType::RemoteIndicesType RemoteIndicesType;

    /** @brief The type of the parallel paradigm we use. */
    typedef typename RemoteIndicesType::ParallelParadigm ParallelParadigm;

    /** @brief The type of the collective communication. */
    typedef typename ParallelParadigm::CollectiveCommunicationType CollectiveCommunicationType;

    /** @brief The type of the map that maps interface information to processors. */
    typedef typename InterfaceType::InformationMap InterfaceMap;

    /** @brief The type of the communicator. */
    typedef typename InterfaceType::CommType CommType;

    /**
     * @brief Constructor.
     * @param interface The interface that defines what indices are to be communicated.
     */
    ThreadCommunicator(InterfaceType& interface);

    /**
     * @brief Send from source to target.
     *
     * The template parameter GatherScatter has to have a static method
     *
     * \code
     * // Gather the data at index index of data
     * static const typename CommPolicy<Data>::IndexedType>& gather(Data& data, int index);
     * // Scatter the value at a index of data
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is SizeOne and
     *
     * \code
     * static const typename CommPolicy<Data>::IndexedType> gather(Data& data, int index, int subindex);
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index, int subindex);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is VariableSize. Here subindex is the subindex of the block at index.
     * @warning The source and target data have to have the same layout as the ones given to the build function in case of variable size values at the indices.
     * @param source The values will be send from here.
     * @param target The received values will be copied to here.
     */
    template<class GatherScatter, class Data>
    void forward(const Data& source, Data& target);

    /**
     * @brief Communicate in the reverse direction, i.e. send from target to source.
     *
     * The template parameter GatherScatter has to have a static method
     *
     * \code
     * // Gather the data at index index of data
     * static const typename CommPolicy<Data>::IndexedType>& gather(Data& data, int index);
     * // Scatter the value at a index of data
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is SizeOne and
     *
     * \code
     * static onst typename CommPolicy<Data>::IndexedType> gather(Data& data, int index, int subindex);
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index, int subindex);
     * \endcode
     * in the case where CommPolicy<Data>::IndexedTypeFlag is VariableSize. Here subindex is the subindex of the block at index.
     * @warning The source and target data have to have the same layout as the ones given to the build function in case of variable size values at the indices.
     * @param source The values will be send from here.
     * @param source The target values will be copied to here.
     */
    template<class GatherScatter, class Data>
    void backward(Data& source, const Data& target);

    /**
     * @brief Forward send where target and source are the same.
     *
     * The template parameter GatherScatter has to have a static method
     *
     * \code
     * // Gather the data at index index of data
     * static const typename CommPolicy<Data>::IndexedType>& gather(Data& data, int index);
     * // Scatter the value at a index of data
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is SizeOne and
     *
     * \code
     * static onst typename CommPolicy<Data>::IndexedType> gather(Data& data, int index, int subindex);
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index, int subindex);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is VariableSize. Here subindex is the subindex of the block at index.
     * @param data Source and target of the communication.
     */
    template<class GatherScatter, class Data>
    void forward(Data& data);

    /**
     * @brief Backward send where target and source are the same.
     *
     * The template parameter GatherScatter has to have a static method
     *
     * \code
     * // Gather the data at index index of data
     * static const typename CommPolicy<Data>::IndexedType>& gather(Data& data, int index);
     * // Scatter the value at a index of data
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is SizeOne and
     *
     * \code
     * static onst typename CommPolicy<Data>::IndexedType> gather(Data& data, int index, int subindex);
     * static void scatter(Data& data, typename CommPolicy<Data>::IndexedType> value, int index, int subindex);
     * \endcode
     *
     * in the case where CommPolicy<Data>::IndexedTypeFlag is VariableSize. Here subindex is the subindex of the block at index.
     * @param data Source and target of the communication.
     */
    template<class GatherScatter, class Data>
    void backward(Data& data);

    /** @brief Destructor. */
    ~ThreadCommunicator()
    {}
  private:
    InterfaceType& interface_;

    /** @brief The interface we currently work with. */
    InterfaceMap& interfaces_;

    CommType communicator_;

    std::vector<int> colors_;

    unsigned int numcolors_;

    /** @brief Compute the coloring scheme.*/
    void computeColoring();

    template<class Data>
    struct InterfaceBuffer
    {
      InterfaceBuffer(const Data* s, Data* t, const InterfaceMap* i) : source(s), target(t), interfaces(i)
      {}

      InterfaceBuffer() : source(nullptr), target(nullptr), interfaces(nullptr)
      {}

      const Data* source;
      Data* target;
      const InterfaceMap* interfaces;
    };

    /** @brief Send and receive Data. */
    template<class GatherScatter, bool FORWARD, class Data>
    void sendRecv(const Data& source, Data& target);
  };

  /** @} */

  template<typename I>
  inline ThreadCommunicator<I>::ThreadCommunicator(InterfaceType& interface) : interface_(interface), interfaces_(interface_.interfaces()), communicator_(interface_.communicator())
  {
    computeColoring();
  }

  template<typename I>
  template<typename GatherScatter,typename Data>
  void ThreadCommunicator<I>::forward(Data& data)
  {
    this->template sendRecv<GatherScatter,true>(data, data);
  }

  template<typename I>
  template<typename GatherScatter, typename Data>
  void ThreadCommunicator<I>::backward(Data& data)
  {
    this->template sendRecv<GatherScatter,false>(data, data);
  }

  template<typename I>
  template<typename GatherScatter, typename Data>
  void ThreadCommunicator<I>::forward(const Data& source, Data& target)
  {
    this->template sendRecv<GatherScatter,true>(source, target);
  }

  template<typename I>
  template<typename GatherScatter, typename Data>
  void ThreadCommunicator<I>::backward(Data& source, const Data& target)
  {
    this->template sendRecv<GatherScatter,false>(target, source);
  }

  template<typename I>
  void ThreadCommunicator<I>::computeColoring()
  {
    RemoteIndicesType& remoteIndices = interface_.remoteIndices();
    ParallelParadigm& parallelParadigm = remoteIndices.parallelParadigm();
    CollectiveCommunicationType& colComm = parallelParadigm.collCommunicator();

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
      colComm.template createBuffer<const std::set<int>*>();
      colComm.template setBuffer<const std::set<int>*>(neighboursPtr, tid);

      typedef typename std::set<int>::iterator SetIterType;
      for(size_t i = 0; i != numThreads; ++i)
      {
        const std::set<int>* ptr = (colComm.template getBuffer<const std::set<int>*>())[i];
        SetIterType itEnd = ptr->end();
        for(SetIterType it = ptr->begin(); it != itEnd; ++it)
          adjMatrix[i][*it] = 1;
      }
      colComm.template deleteBuffer<const std::set<int>*>();

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
  template<typename GatherScatter, bool FORWARD, typename Data>
  void ThreadCommunicator<I>::sendRecv(const Data& source, Data& target)
  {
    RemoteIndicesType& remoteIndices = interface_.remoteIndices();
    ParallelParadigm& parallelParadigm = remoteIndices.parallelParadigm();
    CollectiveCommunicationType& colComm = parallelParadigm.collCommunicator();

    typedef typename InterfaceMap::const_iterator const_iterator;

    const size_t tid = parallelParadigm.threadID();

    // create the buffer to communicate data
    typedef InterfaceBuffer<Data> BufferType;
    colComm.template createBuffer<BufferType>();
    colComm.template setBuffer<BufferType>(BufferType(&source,&target,&interfaces_), tid);

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
            const Data& dest = *(((colComm.template getBuffer<BufferType>())[it->first]).target);
            const_iterator itDest =  (((colComm.template getBuffer<BufferType>())[it->first]).interfaces)->find(tid);

            for(size_t i=0; i < size; i++)
            {
              std::cout<<"gather_"<<GatherScatter::gather(source,it->second.first[i])<<"_tid_"<<tid<<std::endl;
              std::cout<<"scatter_"<<GatherScatter::gather(dest,itDest->second.second[i])<<"_tid_"<<tid<<std::endl;
              //GatherScatter::scatter(dest,GatherScatter::gather(source,it->second.first[i]),itDest->second.second[i]);
            }
          }
        }
        colComm.barrier();
      }
    }
    else
    {
    }

    colComm.template deleteBuffer<BufferType>();

  }

}

#endif
