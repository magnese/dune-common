
// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_THREADCOMMUNICATOR
#define DUNE_THREADCOMMUNICATOR

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
     * @brief Build the buffers and information for the communication process.
     * @param source The source in a forward send. The values will be copied from here to the send buffers.
     * @param target The target in a forward send. The received values will be copied to here.
     */
    //template<class Data>
    //void build(const Data& source, const Data& target);

    /**
     * @brief Send from source to target.
     *
     * The template parameter GatherScatter (e.g. CopyGatherScatter) has to have a static method
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
     * @param source The values will be copied from here to the send buffers.
     * @param dest The received values will be copied to here.
     */
    //template<class GatherScatter, class Data>
    //void forward(const Data& source, Data& dest);

    /**
     * @brief Communicate in the reverse direction, i.e. send from target to source.
     *
     * The template parameter GatherScatter (e.g. CopyGatherScatter) has to have a static method
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
     * @param dest The values will be copied from here to the send buffers.
     * @param source The received values will be copied to here.
     */
    //template<class GatherScatter, class Data>
    //void backward(Data& source, const Data& dest);

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
    //template<class GatherScatter, class Data>
    //void forward(Data& data);

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
    //template<class GatherScatter, class Data>
    //void backward(Data& data);

    /** @brief Free the allocated memory (i.e. buffers and message information. */
    //void free();

    /** @brief Destructor. */
    ~ThreadCommunicator()
    {}
  private:
    InterfaceType& interface_;

    /** @brief The interface we currently work with. */
    InterfaceMap interfaces_;

    CommType communicator_;

    /** @brief Compute the coloring scheme.*/
    void computeColoring();

    /** @brief Send and receive Data. */
    //template<class GatherScatter, bool FORWARD, class Data>
    //void sendRecv(const Data& source, Data& target);
  };

  /** @} */

  template<typename I>
  inline ThreadCommunicator<I>::ThreadCommunicator(InterfaceType& interface) : interface_(interface), interfaces_(interface_.interfaces()), communicator_(interface_.communicator())
  {
    computeColoring();
  }

  template<typename I>
  void ThreadCommunicator<I>::computeColoring()
  {
    RemoteIndicesType& remoteIndices = interface_.remoteIndices();
  }

}

#endif
