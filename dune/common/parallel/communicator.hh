// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_COMMUNICATOR
#define DUNE_COMMUNICATOR

#include "mpicommunicator.hh"
#include <dune/common/exceptions.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/stdstreams.hh>

namespace Dune
{
  /** @defgroup Common_Parallel Parallel Computing based on Indexsets
   * @ingroup ParallelCommunication
   * @brief Provides classes for syncing distributed indexed
   * data structures.
   *
   * In a parallel representation a container \f$x\f$,
   * e.g. a plain C-array, cannot be stored with all entries on each process
   * because of limited memory and efficiency reasons. Therefore
   * it is represented by individual
   * pieces \f$x_p\f$, \f$p=0, \ldots, P-1\f$, where \f$x_p\f$ is the piece stored on
   * process \f$p\f$ of the \f$P\f$ processes participating in the calculation.
   * Although the global representation of the container is not
   * available on any process, a process \f$p\f$ needs to know how the entries
   * of it's local piece \f$x_p\f$ correspond to the entries of the global
   * container \f$x\f$, which would be used in a sequential program. In this
   * module we present classes describing the mapping of the local pieces
   * to the global
   Communicator1* view and the communication interfaces.
   *
   * @section IndexSet Parallel Index Sets
   *
   * Form an abstract point of view a random access container \f$x: I
   * \rightarrow K\f$ provides a
   * mapping from an index set \f$I \subset N_0\f$ onto a set of objects
   * \f$K\f$. Note that we do not require \f$I\f$ to be consecutive. The piece
   * \f$x_p\f$ of the container \f$x\f$ stored on process \f$p\f$ is a mapping \f$x_p:I_p
   * \rightarrow K\f$, where \f$I_p \subset I\f$. Due to efficiency the entries
   * of \f$x_p\f$ should be stored in consecutive memory.
   *
   * This means that for the local computation the data must be addressable
   * by a consecutive index starting from \f$0\f$. When using adaptive
   * discretisation methods there might be the need to reorder the indices
   * after adding and/or deleting some of the discretisation
   * points. Therefore this index does not have to be persistent. Further
   * on we will call this index <em>local index</em>.
   *
   * For the communication phases of our algorithms these locally stored
   * entries must also be addressable by a global identifier to be able to
   * store the received values tagged with the global identifiers at the
   * correct local index in the consecutive local memory chunk. To ease the
   * addition and removal of discretisation points this global identifier has
   * to be persistent. Further on we will call this global identifier
   * <em>global index</em>.
   *
   * Classes to build the mapping are ParallelIndexSet and ParallelLocalIndex.
   * As these just provide a mapping from the global index to the local index,
   * the wrapper class GlobalLookupIndexSet facilitates the reverse lookup.
   *
   * @section remote Remote Index Information
   *
   * To setup communication between the processes every process needs to
   * know what indices are also known to other processes and what
   * attributes are attached to them on the remote side. This information is
   * calculated and encapsulated in class RemoteIndices.
   *
   * @section comm Communication
   *
   * Based on the information about the distributed index sets,  data
   * independent interfaces between different sets of the index sets
   * can be setup using the class Interface.  For the actual communication
   * data dependant communicators can be setup using BufferedCommunicator,
   * DatatypeCommunicator VariableSizeCommunicator based on the interface
   * information. In contrast to the former
   * the latter is independant of the class Interface can work on a map
   * from process number to a pair of index lists describing which local indices
   * are send and received from that processs, respectively.
   */
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Provides utility classes for syncing distributed data.
   * @author Marco Agnese, Markus Blatt
   */

  /** @brief Flag for marking indexed data structures where data at each index is of the same size. */
  struct SizeOne
  {};

  /** @brief Flag for marking indexed data structures where the data at each index may be a variable multiple of another type. */
  struct VariableSize
  {};

  /** @brief Default policy used for communicating an indexed type. */
  template<class V>
  struct CommPolicy
  {
    /**
     * @brief The type the policy is for. It has to provide the mode
     * \code
     * Type::IndexedType operator[](int i);
     * \endcode
     * for the access of the value at index i and a typedef IndexedType. It is assumed that only one entry is at each index (as in scalar vector).
     */
    typedef V Type;

    /** @brief The type we get at each index with operator[]. The default is the value_type typedef of the container. */
    typedef typename V::value_type IndexedType;

    /** @brief Whether the indexed type has variable size or there is always one value at each index. */
    typedef SizeOne IndexedTypeFlag;

    /**
     * @brief Get the address of entry at an index. The default implementation uses operator[] to get the address.
     * @param v An existing representation of the type that has more elements than index.
     * @param index The index of the entry.
     */
    static const void* getAddress(const V& v, int index);

    /** @brief Get the number of primitve elements at that index. The default always returns 1. */
    static int getSize(const V&, int index);
  };

  template<class K, int n> class FieldVector;

  template<class B, class A> class VariableBlockVector;

  template<class K, class A, int n>
  struct CommPolicy<VariableBlockVector<FieldVector<K, n>, A> >
  {
    typedef VariableBlockVector<FieldVector<K, n>, A> Type;

    typedef typename Type::B IndexedType;

    typedef VariableSize IndexedTypeFlag;

    static const void* getAddress(const Type& v, int i);

    static int getSize(const Type& v, int i);
  };

  /** @brief Error thrown if there was a problem with the communication. */
  class CommunicationError : public IOError
  {};

  /** @brief GatherScatter default implementation that just copies data. */
  template<class T>
  struct CopyGatherScatter
  {
    typedef typename CommPolicy<T>::IndexedType IndexedType;

    static const IndexedType& gather(const T& vec, std::size_t i);

    static void scatter(T& vec, const IndexedType& v, std::size_t i);

  };

  /** @brief Communicator interface. It provieds the methods to send and receive data. */
  template<class Imp/*=MPICommunicator*/>
  class Communicator
  {

  public:
    /** @brief The type of the parallel communicator implementation. */
    typedef Imp CommunicatorImplementation;

    /**
     * @brief Constructor.
     * @param interface The interface that defines what indices are to be communicated.
     */
    Communicator();

    /** @brief Build the necessary information for the communication process. */
    template<class Data, class I>
    inline typename enable_if<is_same<SizeOne,typename CommPolicy<Data>::IndexedTypeFlag>::value, void>::type build(const I& interface);

    /**
     * @brief Build the necessary information for the communication process.
     * @param source The source in a forward send.
     * @param target The target in a forward send.
     */
    template<class Data, class I>
    inline void build(const Data& source, const Data& target, const I& interface);

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
    template<class GatherScatter, class Data>
    inline void forward(const Data& source, Data& dest);

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
    template<class GatherScatter, class Data>
    inline void backward(Data& source, const Data& dest);

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
    inline void forward(Data& data);

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
    inline void backward(Data& data);

    /** @brief Free the memory. */
    inline void free();

    /** @brief Destructor. */
    ~Communicator();

  private:
    /** @brief The communicator implementation. */
    CommunicatorImplementation commimp_;
  };

  /** @brief Typedef for compatibility with the old implementation of BufferedCommunicator. */
  //typedef Communicator<> BufferedCommunicator;
#ifndef DOXYGEN

  template<typename V>
  inline const void* CommPolicy<V>::getAddress(const V& v, int index)
  {
    return &(v[index]);
  }

  template<typename V>
  inline int CommPolicy<V>::getSize(const V& v, int index)
  {
    DUNE_UNUSED_PARAMETER(v);
    DUNE_UNUSED_PARAMETER(index);
    return 1;
  }

  template<typename K, typename A, int n>
  inline const void* CommPolicy<VariableBlockVector<FieldVector<K, n>, A> >::getAddress(const Type& v, int index)
  {
    return &(v[index][0]);
  }

  template<typename K, typename A, int n>
  inline int CommPolicy<VariableBlockVector<FieldVector<K, n>, A> >::getSize(const Type& v, int index)
  {
    return v[index].getsize();
  }

  template<typename T>
  inline const typename CopyGatherScatter<T>::IndexedType& CopyGatherScatter<T>::gather(const T & vec, std::size_t i)
  {
    return vec[i];
  }

  template<typename T>
  inline void CopyGatherScatter<T>::scatter(T& vec, const IndexedType& v, std::size_t i)
  {
    vec[i]=v;
  }

  template<typename Imp>
  inline Communicator<Imp>::Communicator() : commimp_()
  {}

  template<typename Imp>
  template<typename Data,typename I>
  inline typename enable_if<is_same<SizeOne, typename CommPolicy<Data>::IndexedTypeFlag>::value, void>::type
    Communicator<Imp>::build(const I& interface)
  {
    return commimp_.template build<Data,I>(interface);
  }

  template<typename Imp>
  template<typename Data,typename I>
  inline void Communicator<Imp>::build(const Data& source, const Data& target, const I& interface)
  {
    commimp_.build(source,target,interface);
  }

  template<typename Imp>
  inline void Communicator<Imp>::free()
  {
    commimp_.free();
  }

  template<typename Imp>
  inline Communicator<Imp>::~Communicator()
  {
    free();
  }

  template<typename Imp>
  template<typename GatherScatter,typename Data>
  inline void Communicator<Imp>::forward(Data& data)
  {
    commimp_.template sendRecv<GatherScatter,true>(data, data);
  }

  template<typename Imp>
  template<typename GatherScatter, typename Data>
  inline void Communicator<Imp>::backward(Data& data)
  {
    commimp_.template sendRecv<GatherScatter,false>(data, data);
  }

  template<typename Imp>
  template<typename GatherScatter, typename Data>
  inline void Communicator<Imp>::forward(const Data& source, Data& dest)
  {
    commimp_.template sendRecv<GatherScatter,true>(source, dest);
  }

  template<typename Imp>
  template<typename GatherScatter, typename Data>
  inline void Communicator<Imp>::backward(Data& source, const Data& dest)
  {
    commimp_.template sendRecv<GatherScatter,false>(dest, source);
  }

#endif  // DOXYGEN

  /** @} */
}

#endif
