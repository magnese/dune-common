// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_MPICOMMUNICATOR
#define DUNE_MPICOMMUNICATOR

#include "remoteindices.hh"
#include "interface.hh"
#include "communicator.hh"
#include <dune/common/exceptions.hh>
#include <dune/common/typetraits.hh>
#include <dune/common/stdstreams.hh>

#if HAVE_MPI
// MPI header
#include <mpi.h>

namespace Dune
{
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Provides the implementation for syncing distributed data via MPI communication.
   * @author Marco Agnese, Markus Blatt
   */

  struct SizeOne;

  struct VariableSize;

  template<class V>
  struct CommPolicy;

  /** @brief Error thrown if there was a problem with the communication. */
  class CommunicationError : public IOError
  {};

  /**
   * @brief An utility class for communicating distributed data structures via MPI datatypes.
   * This communicator creates special MPI datatypes that address the non contiguous elements to be send and received.
   * The idea was to prevent the copying to an additional buffer and the mpi implementation decide whether to allocate buffers or use buffers offered by the
   * interconnection network. Unfortunately the implementation of MPI datatypes seems to be poor.
   */
  template<class T>
  class DatatypeCommunicator : public InterfaceBuilder
  {
    public:
    /** @brief Type of the index set. */
    typedef T ParallelIndexSet;

    /** @brief Type of the underlying remote indices class. */
    typedef Dune::RemoteIndices<ParallelIndexSet> RemoteIndices;

    /** @brief The type of the parallel paradigm we use, e.g. MPIParadigm. */
    typedef typename RemoteIndices::ParallelParadigm ParallelParadigm;

    /** @brief The type of the global index. */
    typedef typename RemoteIndices::GlobalIndex GlobalIndex;

    /** @brief The type of the attribute. */
    typedef typename RemoteIndices::Attribute Attribute;

    /** @brief The type of the local index. */
    typedef typename RemoteIndices::LocalIndex LocalIndex;

    /** @brief Creates a new DatatypeCommunicator. */
    DatatypeCommunicator();

    /** @brief Destructor. */
    ~DatatypeCommunicator();

    /**
     * @brief Builds the interface between the index sets.
     *
     * Has to be called before the actual communication by forward or backward can be called. Nonpublic indices will be ignored!
     *
     * The types T1 and T2 are classes representing a set of enumeration values of type DatatypeCommunicator::Attribute.
     * They have to provide a (static) method
     * \code
     * bool contains(Attribute flag) const;
     * \endcode
     * for checking whether the set contains a specfic flag.
     * This functionality is for example provided the classes EnumItem, EnumRange and Combine.
     *
     * @param remoteIndices The indices present on remote processes.
     * @param sourceFlags The set of attributes which mark indices we send to other processes.
     * @param sendData The indexed data structure whose data will be send.
     * @param destFlags  The set of attributes which mark the indices we might receive values from.
     * @param receiveData The indexed data structure for which we receive data.
     */
    template<class T1, class T2, class V>
    void build(const RemoteIndices& remoteIndices, const T1& sourceFlags, V& sendData, const T2& destFlags, V& receiveData);

    /** @brief Sends the primitive values from the source to the destination. */
    void forward();

    /** @brief Sends the primitive values from the destination to the source. */
    void backward();

    /** @brief Deallocates the MPI requests and data types. */
    void free();
  private:
    enum
    {
      /** @brief Tag for the MPI communication. */
      commTag_ = 234
    };

    /** @brief The indices also known at other processes. */
    const RemoteIndices* remoteIndices_;

    typedef std::map<int,std::pair<MPI_Datatype,MPI_Datatype> > MessageTypeMap;

    /** @brief The datatypes built according to the communication interface. */
    MessageTypeMap messageTypes;

    /** @brief The pointer to the data whose entries we communicate. */
    void* data_;

    MPI_Request* requests_[2];

    /** @brief True if the request and data types were created. */
    bool created_;

    /** @brief Creates the MPI_Requests for the forward communication. */
    template<class V, bool FORWARD>
    void createRequests(V& sendData, V& receiveData);

    /** @brief Creates the data types needed for the unbuffered receive. */
    template<class T1, class T2, class V, bool send>
    void createDataTypes(const T1& source, const T2& destination, V& data);

    /** @brief Initiates the sending and receive. */
    void sendRecv(MPI_Request* req);

    /** @brief Information used for setting up the MPI Datatypes. */
    struct IndexedTypeInformation
    {
      /**
       * @brief Allocate space for setting up the MPI datatype.
       * @param i The number of values the datatype will have.
       */
      void build(int i)
      {
        length = new int[i];
        displ  = new MPI_Aint[i];
        size = i;
      }

      /** @brief Free the allocated space. */
      void free()
      {
        delete[] length;
        delete[] displ;
      }

      /**  @brief The number of values at each index. */
      int* length;

      /** @brief The displacement at each index. */
      MPI_Aint* displ;

      /** @brief The number of elements we send. In case of variable sizes this will differ from size. */
      int elements;

      /** @param The number of indices in the data type.*/
      int size;
    };

    /** @brief Functor for the InterfaceBuilder. It will record the information needed to build the MPI_Datatypes. */
    template<class V>
    struct MPIDatatypeInformation
    {
      /**
       * @brief Constructor.
       * @param data The data we construct an MPI data type for.
       */
      MPIDatatypeInformation(const V& data) : data_(data)
      {}

      /**
       * @brief Reserver space for the information about the datatype.
       * @param proc The rank of the process this information is for.
       * @param size The number of indices the datatype will contain.
       */
      void reserve(int proc, int size)
      {
        information_[proc].build(size);
      }

      /**
       * @brief Add a new index to the datatype.
       * @param proc The rank of the process this index is send to or received from.
       * @param local The index to add.
       */
      void add(int proc, int local)
      {
        IndexedTypeInformation& info=information_[proc];
        assert(info.elements<info.size);
        MPI_Address(const_cast<void*>(CommPolicy<V>::getAddress(data_, local)), info.displ+info.elements);
        info.length[info.elements]=CommPolicy<V>::getSize(data_, local);
        info.elements++;
      }

      /** @brief The information about the datatypes to send to or receive from each process. */
      std::map<int,IndexedTypeInformation> information_;

      /** @brief A representative of the indexed data we send. */
      const V& data_;
    };

  };

  /**
   * @brief An implementation for MPI of the communicator that uses buffers to gather and scatter the data to be send or received.
   *
   * Before the data is sent it it copied to a consecutive buffer and then that buffer is sent.
   * The data is received in another buffer and then copied to the actual position.
   */
  class MPICommunicatorParadigm
  {
  public:
    /** @brief Constructor. */
    MPICommunicatorParadigm();

    /** @brief Build the buffers and information for the communication process. */
    template<class Data, class I>
    typename enable_if<is_same<SizeOne,typename CommPolicy<Data>::IndexedTypeFlag>::value, void>::type build(const I& interface);

    /**
     * @brief Build the buffers and information for the communication process.
     * @param source The source in a forward send. The values will be copied from here to the send buffers.
     * @param target The target in a forward send. The received values will be copied to here.
     */
    template<class Data, class I>
    void build(const Data& source, const Data& target, const I& interface);

    /** @brief Free the allocated memory (i.e. buffers and message information). */
    void free();

    /** @brief Send and receive Data. */
    template<class GatherScatter, bool FORWARD, class Data>
    void sendRecv(const Data& source, Data& target);

    /** @brief Destructor. */
    ~MPICommunicatorParadigm()
    {}

  private:
    /** @brief The type of the map that maps interface information to processors.*/
    typedef std::map<int,std::pair<InterfaceInformation,InterfaceInformation> > InterfaceMap;

    /** @brief Functors for message size caculation. */
    template<class Data, typename IndexedTypeFlag>
    struct MessageSizeCalculator
    {};

    /** @brief Functor for message size caculation for datatypes where at each index is only one value. */
    template<class Data>
    struct MessageSizeCalculator<Data,SizeOne>
    {
      /**
       * @brief Calculate the number of values in message
       * @param info The information about the interface corresponding to the message.
       * @return The number of values in th message.
       */
      inline int operator()(const InterfaceInformation& info) const;

      /**
       * @brief Calculate the number of values in message
       * @param info The information about the interface corresponding to the message.
       * @param data ignored.
       * @return The number of values in th message.
       */
      inline int operator()(const Data& data, const InterfaceInformation& info) const;
    };

    /** @brief Functor for message size caculation for datatypes where at each index can be a variable number of values. */
    template<class Data>
    struct MessageSizeCalculator<Data,VariableSize>
    {
      /**
       * @brief Calculate the number of values in message
       * @param info The information about the interface corresponding to the message.
       * @param data A representative of the data we send.
       * @return The number of values in th message.
       */
      inline int operator()(const Data& data, const InterfaceInformation& info) const;
    };

    /** @brief Functors for message data gathering. */
    template<class Data, class GatherScatter, bool send, typename IndexedTypeFlag>
    struct MessageGatherer
    {};

    /** @brief Functor for message data gathering for datatypes where at each index is only one value. */
    template<class Data, class GatherScatter, bool send>
    struct MessageGatherer<Data,GatherScatter,send,SizeOne>
    {
      /** @brief The type of the values we send. */
      typedef typename CommPolicy<Data>::IndexedType Type;

      /** @brief The type of the functor that does the actual copying during the data Scattering. */
      typedef GatherScatter Gatherer;

      enum
      {
        /** @brief The communication mode. True if this was a forward communication. */
        forward=send
      };

      /**
       * @brief Copies the values to send into the buffer.
       * @param interface The interface used in the send.
       * @param data The data from which we copy the values.
       * @param buffer The send buffer to copy to.
       * @param bufferSize The size of the buffer in bytes. For checks.
       */
      inline void operator()(const InterfaceMap& interface, const Data& data, Type* buffer, size_t bufferSize) const;
    };

    /** @brief Functor for message data scattering for datatypes where at each index can be a variable size of values. */
    template<class Data, class GatherScatter, bool send>
    struct MessageGatherer<Data,GatherScatter,send,VariableSize>
    {
      /** @brief The type of the values we send. */
      typedef typename CommPolicy<Data>::IndexedType Type;

      /** @brief The type of the functor that does the actual copying during the data Scattering. */
      typedef GatherScatter Gatherer;

      enum
      {
        /** @brief The communication mode. True if this was a forward communication. */
        forward=send
      };

      /**
       * @brief Copies the values to send into the buffer.
       * @param interface The interface used in the send.
       * @param data The data from which we copy the values.
       * @param buffer The send buffer to copy to.
       * @param bufferSize The size of the buffer in bytes. For checks.
       */
      inline void operator()(const InterfaceMap& interface, const Data& data, Type* buffer, size_t bufferSize) const;
    };

    /** @brief Functors for message data scattering. */
    template<class Data, class GatherScatter, bool send, typename IndexedTypeFlag>
    struct MessageScatterer
    {};

    /** @brief Functor for message data gathering for datatypes where at each index is only one value. */
    template<class Data, class GatherScatter, bool send>
    struct MessageScatterer<Data,GatherScatter,send,SizeOne>
    {
      /** @brief The type of the values we send. */
      typedef typename CommPolicy<Data>::IndexedType Type;

      /** @brief The type of the functor that does the actual copying during the data Scattering. */
      typedef GatherScatter Scatterer;

      enum
      {
        /** @brief The communication mode. True if this was a forward communication. */
        forward=send
      };

      /**
       * @brief Copy the message data from the receive buffer to the data.
       * @param interface The interface used in the send.
       * @param data The data to which we copy the values.
       * @param buffer The receive buffer to copy from.
       * @param proc The rank of the process the message is from.
       */
      inline void operator()(const InterfaceMap& interface, Data& data, Type* buffer, const int& proc) const;
    };

    /** @brief Functor for message data scattering for datatypes where at each index can be a variable size of values. */
    template<class Data, class GatherScatter, bool send>
    struct MessageScatterer<Data,GatherScatter,send,VariableSize>
    {
      /** @brief The type of the values we send. */
      typedef typename CommPolicy<Data>::IndexedType Type;

      /** @brief The type of the functor that does the actual copying during the data Scattering. */
      typedef GatherScatter Scatterer;

      enum
      {
        /** @brief The communication mode. True if this was a forward communication. */
        forward=send
      };

      /**
       * @brief Copy the message data from the receive buffer to the data.
       * @param interface The interface used in the send.
       * @param data The data to which we copy the values.
       * @param buffer The receive buffer to copy from.
       * @param proc The rank of the process the message is from.
       */
      inline void operator()(const InterfaceMap& interface, Data& data, Type* buffer, const int& proc) const;
    };

    /** @brief Information about a message to send. */
    struct MessageInformation
    {
      /** @brief Constructor. */
      MessageInformation() : start_(0), size_(0)
      {}

      /**
       * @brief Constructor.
       * @param start The start of the message in the global buffer. Not in bytes but in number of values from the beginning of the buffer.
       * @param size The size of the message in bytes.
       */
      MessageInformation(size_t start, size_t size) : start_(start), size_(size)
      {}

      /** @brief Start of the message in the buffer counted in number of value. */
      size_t start_;

      /** @brief Number of bytes in the message. */
      size_t size_;
    };

    /**
     * @brief Type of the map of information about the messages to send.
     * The key is the process number to communicate with and the value is the pair of information about sending and receiving messages.
     */
    typedef std::map<int,std::pair<MessageInformation,MessageInformation> > InformationMap;

    /** @brief Gathered information about the messages to send. */
    InformationMap messageInformation_;

    /** @brief Communication buffers. */
    char* buffers_[2];

    /** @brief The size of the communication buffers. */
    size_t bufferSize_[2];

    enum
    {
      /** @brief The tag we use for communication. */
      commTag_
    };

    /** @brief The interface we currently work with. */
    InterfaceMap interfaces_;

    /** @brief The communicator. */
    MPI_Comm communicator_;
  };

#ifndef DOXYGEN

  template<typename T>
  DatatypeCommunicator<T>::DatatypeCommunicator() : remoteIndices_(0), created_(false)
  {
    requests_[0]=0;
    requests_[1]=0;
  }

  template<typename T>
  DatatypeCommunicator<T>::~DatatypeCommunicator()
  {
    free();
  }

  template<typename T>
  template<class T1, class T2, class V>
  inline void DatatypeCommunicator<T>::build(const RemoteIndices& remoteIndices, const T1& source, V& sendData, const T2& destination, V& receiveData)
  {
    remoteIndices_ = &remoteIndices;
    free();
    createDataTypes<T1,T2,V,false>(source, destination, receiveData);
    createDataTypes<T1,T2,V,true>(source, destination, sendData);
    createRequests<V,true>(sendData, receiveData);
    createRequests<V,false>(receiveData, sendData);
    created_=true;
  }

  template<typename T>
  void DatatypeCommunicator<T>::free()
  {
    if(created_)
    {
      delete[] requests_[0];
      delete[] requests_[1];
      typedef MessageTypeMap::iterator iterator;
      typedef MessageTypeMap::const_iterator const_iterator;

      const const_iterator end=messageTypes.end();

      for(iterator process = messageTypes.begin(); process != end; ++process)
      {
        MPI_Datatype *type = &(process->second.first);
        int finalized=0;
#if MPI_2
        MPI_Finalized(&finalized);
#endif
        if(*type!=MPI_DATATYPE_NULL && !finalized)
          MPI_Type_free(type);
        type = &(process->second.second);
        if(*type!=MPI_DATATYPE_NULL && !finalized)
          MPI_Type_free(type);
      }
      messageTypes.clear();
      created_=false;
    }
  }

  template<typename T>
  template<class T1, class T2, class V, bool send>
  void DatatypeCommunicator<T>::createDataTypes(const T1& sourceFlags, const T2& destFlags, V& data)
  {
    MPIDatatypeInformation<V>  dataInfo(data);
    this->template buildInterface<RemoteIndices,T1,T2,MPIDatatypeInformation<V>,send>(*remoteIndices_,sourceFlags, destFlags, dataInfo);

    typedef typename RemoteIndices::RemoteIndexMap::const_iterator const_iterator;
    const const_iterator end=this->remoteIndices_->end();

    // allocate MPI_Datatypes and deallocate memory for the type construction
    for(const_iterator process=this->remoteIndices_->begin(); process != end; ++process)
    {
      IndexedTypeInformation& info=dataInfo.information_[process->first];
      // shift the displacement
      MPI_Aint base;
      MPI_Address(const_cast<void *>(CommPolicy<V>::getAddress(data, 0)), &base);

      for(int i=0; i< info.elements; i++)
        info.displ[i]-=base;

      // create data type
      MPI_Datatype* type = &( send ? messageTypes[process->first].first : messageTypes[process->first].second);
      MPI_Type_hindexed(info.elements, info.length, info.displ, MPITraits<typename CommPolicy<V>::IndexedType>::getType(), type);
      MPI_Type_commit(type);
      // deallocate memory
      info.free();
    }
  }

  template<typename T>
  template<class V, bool createForward>
  void DatatypeCommunicator<T>::createRequests(V& sendData, V& receiveData)
  {
    typedef std::map<int,std::pair<MPI_Datatype,MPI_Datatype> >::const_iterator MapIterator;
    static int index = createForward ? 1 : 0;
    int noMessages = messageTypes.size();
    // allocate request handles
    requests_[index] = new MPI_Request[2*noMessages];
    const MapIterator end = messageTypes.end();
    int request=0;

    // set up the requests for receiving first
    for(MapIterator process = messageTypes.begin(); process != end; ++process, ++request)
    {
      MPI_Datatype type = createForward ? process->second.second : process->second.first;
      void* address = const_cast<void*>(CommPolicy<V>::getAddress(receiveData,0));
      MPI_Recv_init(address, 1, type, process->first, commTag_, this->remoteIndices_->communicator(), requests_[index]+request);
    }

    // and now the send requests
    for(MapIterator process = messageTypes.begin(); process != end; ++process, ++request)
    {
      MPI_Datatype type = createForward ? process->second.first : process->second.second;
      void* address =  const_cast<void*>(CommPolicy<V>::getAddress(sendData, 0));
      MPI_Ssend_init(address, 1, type, process->first, commTag_, this->remoteIndices_->communicator(), requests_[index]+request);
    }
  }

  template<typename T>
  void DatatypeCommunicator<T>::forward()
  {
    sendRecv(requests_[1]);
  }

  template<typename T>
  void DatatypeCommunicator<T>::backward()
  {
    sendRecv(requests_[0]);
  }

  template<typename T>
  void DatatypeCommunicator<T>::sendRecv(MPI_Request* requests)
  {
    int noMessages = messageTypes.size();
    // start the receive calls first
    MPI_Startall(noMessages, requests);
    // now the send calls
    MPI_Startall(noMessages, requests+noMessages);

    // wait for completion of the communication send first then receive
    MPI_Status* status=new MPI_Status[2*noMessages];
    for(int i=0; i<2*noMessages; i++)
      status[i].MPI_ERROR=MPI_SUCCESS;

    int send = MPI_Waitall(noMessages, requests+noMessages, status+noMessages);
    int receive = MPI_Waitall(noMessages, requests, status);

    // error checks
    int success=1, globalSuccess=0;
    if(send==MPI_ERR_IN_STATUS)
    {
      int rank;
      MPI_Comm_rank(this->remoteIndices_->communicator(), &rank);
      std::cerr<<rank<<": Error in sending :"<<std::endl;
      // search for the error
      for(int i=noMessages; i< 2*noMessages; i++)
        if(status[i].MPI_ERROR!=MPI_SUCCESS)
        {
          char message[300];
          int messageLength;
          MPI_Error_string(status[i].MPI_ERROR, message, &messageLength);
          std::cerr<<" source="<<status[i].MPI_SOURCE<<" message: ";
          for(int i=0; i< messageLength; i++)
            std::cout<<message[i];
        }
      std::cerr<<std::endl;
      success=0;
    }

    if(receive==MPI_ERR_IN_STATUS)
    {
      int rank;
      MPI_Comm_rank(this->remoteIndices_->communicator(), &rank);
      std::cerr<<rank<<": Error in receiving!"<<std::endl;
      // search for the error
      for(int i=0; i< noMessages; i++)
        if(status[i].MPI_ERROR!=MPI_SUCCESS)
        {
          char message[300];
          int messageLength;
          MPI_Error_string(status[i].MPI_ERROR, message, &messageLength);
          std::cerr<<" source="<<status[i].MPI_SOURCE<<" message: ";
          for(int i=0; i< messageLength; i++)
            std::cerr<<message[i];
        }
      std::cerr<<std::endl;
      success=0;
    }

    MPI_Allreduce(&success, &globalSuccess, 1, MPI_INT, MPI_MIN, this->remoteIndices_->communicator());

    delete[] status;

    if(!globalSuccess)
      DUNE_THROW(CommunicationError, "A communication error occurred!");
  }

  inline MPICommunicatorParadigm::MPICommunicatorParadigm()
  {
    buffers_[0]=0;
    buffers_[1]=0;
    bufferSize_[0]=0;
    bufferSize_[1]=0;
  }

  template<typename Data,typename I>
  typename enable_if<is_same<SizeOne, typename CommPolicy<Data>::IndexedTypeFlag>::value, void>::type
    MPICommunicatorParadigm::build(const I& interface)
  {
    communicator_ = interface.communicator();
    interfaces_ = interface.interfaces();
    typedef typename CommPolicy<Data>::IndexedTypeFlag Flag;
    typedef typename CommPolicy<Data>::IndexedType IndexedType;
    const size_t indexedTypeSize = sizeof(IndexedType);

    bufferSize_[0]=0;
    bufferSize_[1]=0;

    typedef typename I::InformationMap::const_iterator const_iterator;
    const const_iterator end = interfaces_.end();
    for(const_iterator interfacePair = interfaces_.begin(); interfacePair != end; ++interfacePair)
    {
      int noSend = MessageSizeCalculator<Data,Flag>() (interfacePair->second.first);
      int noRecv = MessageSizeCalculator<Data,Flag>() (interfacePair->second.second);
      if(noSend + noRecv > 0)
        messageInformation_.insert(std::make_pair(interfacePair->first,
                                                std::make_pair(MessageInformation(bufferSize_[0], noSend*indexedTypeSize),
                                                               MessageInformation(bufferSize_[1], noRecv*indexedTypeSize))));
      bufferSize_[0] += noSend;
      bufferSize_[1] += noRecv;
    }

    // allocate the buffers
    bufferSize_[0] *= indexedTypeSize;
    bufferSize_[1] *= indexedTypeSize;

    buffers_[0] = new char[bufferSize_[0]];
    buffers_[1] = new char[bufferSize_[1]];
  }

  template<typename Data,typename I>
  void MPICommunicatorParadigm::build(const Data& source, const Data& target, const I& interface)
  {
    communicator_ = interface.communicator();
    interfaces_ =interface.interfaces();
    typedef typename CommPolicy<Data>::IndexedTypeFlag Flag;
    typedef typename CommPolicy<Data>::IndexedType IndexedType;
    const size_t indexedTypeSize = sizeof(IndexedType);

    bufferSize_[0]=0;
    bufferSize_[1]=0;

    typedef typename I::InformationMap::const_iterator const_iterator;
    const const_iterator end = interfaces_.end();
    for(const_iterator interfacePair = interfaces_.begin(); interfacePair != end; ++interfacePair)
    {
      int noSend = MessageSizeCalculator<Data,Flag>() (source, interfacePair->second.first);
      int noRecv = MessageSizeCalculator<Data,Flag>() (target, interfacePair->second.second);
      if(noSend + noRecv > 0)
        messageInformation_.insert(std::make_pair(interfacePair->first,
                                                std::make_pair(MessageInformation(bufferSize_[0], noSend*indexedTypeSize),
                                                               MessageInformation(bufferSize_[1], noRecv*indexedTypeSize))));
      bufferSize_[0] += noSend;
      bufferSize_[1] += noRecv;
    }

    // allocate the buffers
    bufferSize_[0] *= indexedTypeSize;
    bufferSize_[1] *= indexedTypeSize;

    buffers_[0] = new char[bufferSize_[0]];
    buffers_[1] = new char[bufferSize_[1]];
  }

  inline void MPICommunicatorParadigm::free()
  {
    messageInformation_.clear();
    if(buffers_[0])
      delete[] buffers_[0];

    if(buffers_[1])
      delete[] buffers_[1];
    buffers_[0]=buffers_[1]=0;
  }

  template<typename Data>
  inline int MPICommunicatorParadigm::MessageSizeCalculator<Data,SizeOne>::operator()(const InterfaceInformation& info) const
  {
    return info.size();
  }

  template<typename Data>
  inline int MPICommunicatorParadigm::MessageSizeCalculator<Data,SizeOne>::operator()(const Data&, const InterfaceInformation& info) const
  {
    return operator()(info);
  }

  template<typename Data>
  inline int MPICommunicatorParadigm::MessageSizeCalculator<Data, VariableSize>::operator()(const Data& data, const InterfaceInformation& info) const
  {
    int entries=0;
    for(size_t i=0; i < info.size(); i++)
      entries += CommPolicy<Data>::getSize(data,info[i]);
    return entries;
  }

  template<typename Data, typename GatherScatter, bool FORWARD>
  inline void MPICommunicatorParadigm::MessageGatherer<Data,GatherScatter,FORWARD,VariableSize>::operator()(const InterfaceMap& interfaces,const Data& data, Type* buffer, size_t bufferSize) const
  {
#ifdef DUNE_ISTL_WITH_CHECKING
    typedef typename CommPolicy<Data>::IndexedType IndexedType;
    const size_t indexedTypeSize = sizeof(IndexedType);
#endif
    typedef typename InterfaceMap::const_iterator const_iterator;

    const const_iterator end = interfaces.end();
    size_t index=0;

    for(const_iterator interfacePair = interfaces.begin(); interfacePair != end; ++interfacePair)
    {
      int size = forward ? interfacePair->second.first.size() : interfacePair->second.second.size();
      for(int i=0; i < size; i++)
      {
        int local = forward ? interfacePair->second.first[i] : interfacePair->second.second[i];
        for(std::size_t j=0; j < CommPolicy<Data>::getSize(data, local); j++, index++)
        {
#ifdef DUNE_ISTL_WITH_CHECKING
          assert(bufferSize>=(index+1)*indexedTypeSize);
#endif
          buffer[index]=GatherScatter::gather(data, local, j);
        }
      }
    }

  }

  template<typename Data, typename GatherScatter, bool FORWARD>
  inline void MPICommunicatorParadigm::MessageGatherer<Data,GatherScatter,FORWARD,SizeOne>::operator()(const InterfaceMap& interfaces, const Data& data, Type* buffer, size_t bufferSize) const
  {
#ifdef DUNE_ISTL_WITH_CHECKING
    typedef typename CommPolicy<Data>::IndexedType IndexedType;
    const size_t indexedTypeSize = sizeof(IndexedType);
#endif
    DUNE_UNUSED_PARAMETER(bufferSize);
    typedef typename InterfaceMap::const_iterator const_iterator;
    const const_iterator end = interfaces.end();
    size_t index = 0;

    for(const_iterator interfacePair = interfaces.begin(); interfacePair != end; ++interfacePair)
    {
      size_t size = FORWARD ? interfacePair->second.first.size() : interfacePair->second.second.size();
      for(size_t i=0; i < size; i++)
      {
#ifdef DUNE_ISTL_WITH_CHECKING
        assert(bufferSize>=(index+1)*indexedTypeSize);
#endif
        buffer[index++] = GatherScatter::gather(data, FORWARD ? interfacePair->second.first[i] : interfacePair->second.second[i]);
      }
    }

  }

  template<typename Data, typename GatherScatter, bool FORWARD>
  inline void MPICommunicatorParadigm::MessageScatterer<Data,GatherScatter,FORWARD,VariableSize>::operator()(const InterfaceMap& interfaces, Data& data, Type* buffer, const int& proc) const
  {
    typedef typename InterfaceMap::value_type::second_type::first_type Information;
    const typename InterfaceMap::const_iterator infoPair = interfaces.find(proc);

    assert(infoPair!=interfaces.end());

    const Information& info = FORWARD ? infoPair->second.second : infoPair->second.first;
    for(size_t i=0, index=0; i < info.size(); i++)
    {
      for(size_t j=0; j < CommPolicy<Data>::getSize(data, info[i]); j++)
        GatherScatter::scatter(data, buffer[index++], info[i], j);
    }

  }

  template<typename Data, typename GatherScatter, bool FORWARD>
  inline void MPICommunicatorParadigm::MessageScatterer<Data,GatherScatter,FORWARD,SizeOne>::operator()(const InterfaceMap& interfaces, Data& data, Type* buffer, const int& proc) const
  {
    typedef typename InterfaceMap::value_type::second_type::first_type Information;
    const typename InterfaceMap::const_iterator infoPair = interfaces.find(proc);

    assert(infoPair!=interfaces.end());

    const Information& info = FORWARD ? infoPair->second.second : infoPair->second.first;
    for(size_t i=0; i < info.size(); i++)
      GatherScatter::scatter(data, buffer[i], info[i]);
  }

  template<typename GatherScatter, bool FORWARD, typename Data>
  void MPICommunicatorParadigm::sendRecv(const Data& source, Data& dest)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    typedef typename CommPolicy<Data>::IndexedTypeFlag Flag;
    typedef typename CommPolicy<Data>::IndexedType IndexedType;

    IndexedType *sendBuffer, *recvBuffer;
    size_t sendBufferSize;
#ifndef NDEBUG
    size_t recvBufferSize;
#endif

    if(FORWARD)
    {
      sendBuffer = reinterpret_cast<IndexedType*>(buffers_[0]);
      sendBufferSize = bufferSize_[0];
      recvBuffer = reinterpret_cast<IndexedType*>(buffers_[1]);
#ifndef NDEBUG
      recvBufferSize = bufferSize_[1];
#endif
    }
    else
    {
      sendBuffer = reinterpret_cast<IndexedType*>(buffers_[1]);
      sendBufferSize = bufferSize_[1];
      recvBuffer = reinterpret_cast<IndexedType*>(buffers_[0]);
#ifndef NDEBUG
      recvBufferSize = bufferSize_[0];
#endif
    }

    MessageGatherer<Data,GatherScatter,FORWARD,Flag>() (interfaces_, source, sendBuffer, sendBufferSize);

    MPI_Request* sendRequests = new MPI_Request[messageInformation_.size()];
    MPI_Request* recvRequests = new MPI_Request[messageInformation_.size()];

    // setup receive first
    typedef typename InformationMap::const_iterator const_iterator;

    const const_iterator end = messageInformation_.end();
    size_t i=0;
    int* processMap = new int[messageInformation_.size()];

    for(const_iterator info = messageInformation_.begin(); info != end; ++info, ++i)
    {
      processMap[i]=info->first;
      if(FORWARD)
      {
        assert(info->second.second.start_*sizeof(IndexedType)+info->second.second.size_ <= recvBufferSize );
        Dune::dvverb<<rank<<": receiving "<<info->second.second.size_<<" from "<<info->first<<std::endl;
        MPI_Irecv(recvBuffer+info->second.second.start_, info->second.second.size_, MPI_BYTE, info->first, commTag_, communicator_, recvRequests+i);
      }
      else
      {
        assert(info->second.first.start_*sizeof(IndexedType)+info->second.first.size_ <= recvBufferSize );
        Dune::dvverb<<rank<<": receiving "<<info->second.first.size_<<" to "<<info->first<<std::endl;
        MPI_Irecv(recvBuffer+info->second.first.start_, info->second.first.size_, MPI_BYTE, info->first, commTag_, communicator_, recvRequests+i);
      }
    }

    // now the send requests
    i=0;
    for(const_iterator info = messageInformation_.begin(); info != end; ++info, ++i)
      if(FORWARD)
      {
        assert(info->second.second.start_*sizeof(IndexedType)+info->second.second.size_ <= recvBufferSize );
        Dune::dvverb<<rank<<": sending "<<info->second.first.size_<<" to "<<info->first<<std::endl;
        assert(info->second.first.start_*sizeof(IndexedType)+info->second.first.size_ <= sendBufferSize );
        MPI_Issend(sendBuffer+info->second.first.start_, info->second.first.size_, MPI_BYTE, info->first, commTag_, communicator_, sendRequests+i);
      }
      else
      {
        assert(info->second.second.start_*sizeof(IndexedType)+info->second.second.size_ <= sendBufferSize );
        Dune::dvverb<<rank<<": sending "<<info->second.second.size_<<" to "<<info->first<<std::endl;
        MPI_Issend(sendBuffer+info->second.second.start_, info->second.second.size_, MPI_BYTE, info->first, commTag_, communicator_, sendRequests+i);
      }

    // wait for completion of receive and immediately start scatter
    i=0;
    int finished = MPI_UNDEFINED;
    MPI_Status status;

    for(i=0; i< messageInformation_.size(); i++)
    {
      status.MPI_ERROR=MPI_SUCCESS;
      MPI_Waitany(messageInformation_.size(), recvRequests, &finished, &status);
      assert(finished != MPI_UNDEFINED);

      if(status.MPI_ERROR==MPI_SUCCESS)
      {
        int& proc = processMap[finished];
        typename InformationMap::const_iterator infoIter = messageInformation_.find(proc);
        assert(infoIter != messageInformation_.end());

        MessageInformation info = (FORWARD) ? infoIter->second.second : infoIter->second.first;
        assert(info.start_+info.size_ <= recvBufferSize);

        MessageScatterer<Data,GatherScatter,FORWARD,Flag>() (interfaces_, dest, recvBuffer+info.start_, proc);
      }
      else
        std::cerr<<rank<<": MPI_Error occurred while receiving message from "<<processMap[finished]<<std::endl;
    }

    MPI_Status recvStatus;

    // wait for completion of sends
    for(i=0; i< messageInformation_.size(); i++)
      if(MPI_SUCCESS!=MPI_Wait(sendRequests+i, &recvStatus))
        std::cerr<<rank<<": MPI_Error occurred while sending message to "<<processMap[finished]<<std::endl;

    delete[] processMap;
    delete[] sendRequests;
    delete[] recvRequests;
  }

#endif  // DOXYGEN

  /** @} */
}

#else
/** @brief Class needed when MPI is not defined. */
namespace Dune
{
struct MPICommunicatorParadigm{};
}
#endif

#endif
