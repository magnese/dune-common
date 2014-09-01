// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_INTERFACE_HH
#define DUNE_INTERFACE_HH

#include "remoteindices.hh"
#include <dune/common/enumset.hh>

namespace Dune
{
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Provides classes for building the communication interface between remote indices.
   * @author Marco Agnese, Markus Blatt
   */

  /** @} */
  /**
   * @brief Base class of all classes representing a communication interface.
   *
   * It provides an generic utility method for building the interface for a set of remote indices.
   */
  class InterfaceBuilder
  {
  public:
    class RemoteIndicesStateError : public InvalidStateException
    {};

    virtual ~InterfaceBuilder()
    {}

  protected:
    /** @brief Not for public use. */
    InterfaceBuilder()
    {}

    /**
     * @brief Builds the interface between remote processes.
     *
     *
     * The types T1 and T2 are classes representing a set of enumeration values of type InterfaceBuilder::Attribute. They have to provide a (static) method
     * \code
     * bool contains(Attribute flag) const;
     * \endcode
     * for checking whether the set contains a specfic flag. This functionality is for example provided the classes EnumItem, EnumRange and Combine.
     *
     * If the template parameter send is true the sending side of the interface will be built, otherwise the information for receiving will be built.
     *
     *
     * If the template parameter send is true we create interface for sending in a forward communication.
     *
     * @param remoteIndices The indices known to remote processes.
     * @param sourceFlags The set of flags marking source indices.
     * @param destFlags The setof flags markig destination indices.
     * @param functor A functor for callbacks. It should provide the following methods:
     * \code
     * // Reserve memory for the interface to processor proc. The interface has to hold size entries
     * void reserve(int proc, int size);
     * // Add an entry to the interface. We will send/receive size entries at index local to process proc
     * void add(int proc, int local);
     * \endcode
     */
    template<class R, class T1, class T2, class Op, bool send>
    void buildInterface (const R& remoteIndices, const T1& sourceFlags, const T2& destFlags, Op& functor) const;
  };

  /**
   * @brief Information describing an interface.
   *
   * This class is used for temporary gathering information about the interface needed for actually building it.
   * It is used be class Interface as functor for InterfaceBuilder::build.
   */
  class InterfaceInformation
  {

  public:

    /** @brief Get the number of entries in the interface. */
    size_t size() const
    {
      return size_;
    }

    /**
     * @brief Get the local index for an entry.
     * @param i The  index of the entry.
     */
    std::size_t& operator[](size_t i)
    {
      assert(i<size_);
      return indices_[i];
    }

    /**
     * @brief Get the local index for an entry.
     * @param i The  index of the entry.
     */
    std::size_t operator[](size_t i) const
    {
      assert(i<size_);
      return indices_[i];
    }

    /**
     * @brief Reserve space for a number of entries.
     * @param size The maximum number of entries to hold.
     */
    void reserve(size_t size)
    {
      indices_ = new std::size_t[size];
      maxSize_ = size;
    }

    /** brief Frees allocated memory. */
    void free()
    {
      if(indices_)
        delete[] indices_;
      maxSize_ = 0;
      size_=0;
      indices_=0;
    }

    /** @brief Add a new index to the interface. */
    void add(std::size_t index)
    {
      assert(size_<maxSize_);
      indices_[size_++]=index;
    }

    InterfaceInformation() : size_(0), maxSize_(0), indices_(0)
    {}

    virtual ~InterfaceInformation()
    {}

    bool operator!=(const InterfaceInformation& o) const
    {
      return !operator==(o);
    }

    bool operator==(const InterfaceInformation& o) const
    {
      if(size_!=o.size_)
        return false;
      for(std::size_t i=0; i< size_; ++i)
        if(indices_[i]!=o.indices_[i])
          return false;
      return true;
    }

  private:
    /** @brief The number of entries in the interface. */
    size_t size_;

    /** @brief The maximum number of indices we can hold. */
    size_t maxSize_;

    /** @brief The local indices of the interface. */
    std::size_t* indices_;
  };

  /** @addtogroup Common_Parallel
   *
   * @{
   */

  /**
   * @brief Communication interface between remote and local indices.
   *
   * Describes the communication interface between indices on the local process and those on remote processes.
   */
  template<class R>
  class Interface : public InterfaceBuilder
  {

  public:
    /** @brief The type of the map form process number to InterfaceInformation for sending and receiving to and from it. */
    typedef std::map<int,std::pair<InterfaceInformation,InterfaceInformation> > InformationMap;

    /** @brief The type of remote indices. */
    typedef R RemoteIndicesType;

    /** @brief The type of the communicator. */
    typedef typename RemoteIndicesType::CommType CommType;

    /**
     * @brief Builds the interface.
     *
     * The types T1 and T2 are classes representing a set of enumeration values of type Interface::Attribute. They have to provide a (static) method
     * \code
     * bool contains(Attribute flag) const;
     * \endcode
     * for checking whether the set contains a specfic flag. This functionality is for example provided the classes EnumItem, EnumRange and Combine.
     * @param sourceFlags The set of flags marking indices we send from.
     * @param destFlags The set of flags marking indices we receive for.
     */
    template<typename T1, typename T2>
    void build(const T1& sourceFlags, const T2& destFlags);

    /** @brief Frees memory allocated during the build. */
    void free();

    /** @brief Get the communicator. */
    CommType communicator() const;

    /**
     * @brief Get information about the interfaces.
     *
     * @return Map of the interfaces.
     * The key of the map is the process number and the value is the information pair (first the send and then the receive information).
     */
    const InformationMap& interfaces() const;

    /**
     * @brief Get information about the interfaces.
     *
     * @return Map of the interfaces.
     * The key of the map is the process number and the value is the information pair (first the send and then the receive information).
     */
    InformationMap& interfaces();

    Interface(RemoteIndicesType& remoteIndices) : remoteindices_(remoteIndices), communicator_(remoteindices_.communicator()), interfaces_()
    {}

    /** @brief Print the interface to std::out for debugging. */
    void print() const;

    bool operator!=(const Interface& o) const
    {
      return ! operator==(o);
    }

    bool operator==(const Interface& o) const
    {
      if(communicator_!=o.communicator_)
        return false;
      if(interfaces_.size()!=o.interfaces_.size())
        return false;
      typedef InformationMap::const_iterator MIter;

      for(MIter m=interfaces_.begin(), om=o.interfaces_.begin();
          m!=interfaces_.end(); ++m, ++om)
      {
        if(om->first!=m->first)
          return false;
        if(om->second.first!=om->second.first)
          return false;
        if(om->second.second!=om->second.second)
          return false;
      }
      return true;
    }

    /** @brief Destructor. */
    virtual ~Interface();

    void strip();
  protected:
    RemoteIndicesType& remoteindices_;

    /** @brief The communicator we use. */
    CommType communicator_;
  private:
    /**
     * @brief Information about the interfaces.
     *
     * The key of the map is the process number and the value is the information pair (first the send and then the receive information).
     */
    InformationMap interfaces_;

    template<bool send>
    class InformationBuilder
    {
    public:
      InformationBuilder(InformationMap& interfaces) : interfaces_(interfaces)
      {}

      void reserve(int proc, int size)
      {
        if(send)
          interfaces_[proc].first.reserve(size);
        else
          interfaces_[proc].second.reserve(size);
      }

      void add(int proc, std::size_t local)
      {
        if(send)
          interfaces_[proc].first.add(local);
        else
          interfaces_[proc].second.add(local);
      }

    private:
      InformationMap& interfaces_;
    };
  };

  template<class R, class T1, class T2, class Op, bool send>
  void InterfaceBuilder::buildInterface(const R& remoteIndices, const T1& sourceFlags, const T2& destFlags, Op& interfaceInformation) const
  {

    if(!remoteIndices.isSynced())
      DUNE_THROW(RemoteIndicesStateError,"RemoteIndices is not in sync with the index set. Call RemoteIndices::rebuild first!");
    // allocate the memory for the data type construction
    typedef R RemoteIndices;
    typedef typename RemoteIndices::RemoteIndexMap::const_iterator const_iterator;

    const const_iterator end=remoteIndices.end();

    // allocate memory for the type construction
    for(const_iterator process=remoteIndices.begin(); process != end; ++process)
    {
      // messure the number of indices send to the remote process first
      int size=0;
      typedef typename RemoteIndices::RemoteIndexList::const_iterator RemoteIterator;
      const RemoteIterator remoteEnd = send ? process->second.first->end() : process->second.second->end();
      RemoteIterator remote = send ? process->second.first->begin() : process->second.second->begin();

      while(remote!=remoteEnd)
      {
        if( send ? destFlags.contains(remote->attribute()) : sourceFlags.contains(remote->attribute()))
        {
          // do we send the index?
          if( send ? sourceFlags.contains(remote->localIndexPair().local().attribute()) : destFlags.contains(remote->localIndexPair().local().attribute()))
            ++size;
        }
        ++remote;
      }
      interfaceInformation.reserve(process->first, size);
    }

    // compare the local and remote indices and set up the types
    for(const_iterator process=remoteIndices.begin(); process != end; ++process)
    {
      typedef typename RemoteIndices::RemoteIndexList::const_iterator RemoteIterator;
      const RemoteIterator remoteEnd = send ? process->second.first->end() : process->second.second->end();
      RemoteIterator remote = send ? process->second.first->begin() : process->second.second->begin();

      while(remote!=remoteEnd)
      {
        if( send ? destFlags.contains(remote->attribute()) : sourceFlags.contains(remote->attribute()))
        {
          // do we send the index?
          if( send ? sourceFlags.contains(remote->localIndexPair().local().attribute()) : destFlags.contains(remote->localIndexPair().local().attribute()))
            interfaceInformation.add(process->first,remote->localIndexPair().local().local());
        }
        ++remote;
      }
    }
  }

  template<typename R>
  inline typename Interface<R>::CommType Interface<R>::communicator() const
  {
    return communicator_;
  }

  template<typename R>
  inline const typename Interface<R>::InformationMap& Interface<R>::interfaces() const
  {
    return interfaces_;
  }

  template<typename R>
  inline typename Interface<R>::InformationMap& Interface<R>::interfaces()
  {
    return interfaces_;
  }

  template<typename R>
  inline void Interface<R>::print() const
  {
    typedef InformationMap::const_iterator const_iterator;
    const const_iterator end=interfaces_.end();

    for(const_iterator infoPair=interfaces_.begin(); infoPair!=end; ++infoPair)
    {
      {
        std::cout<<"send for process "<<infoPair->first<<": ";
        const InterfaceInformation& info(infoPair->second.first);
        for(size_t i=0; i < info.size(); i++)
          std::cout<<info[i]<<" ";
        std::cout<<std::endl;
      }
      {
        std::cout<<"receive for process "<<infoPair->first<<": ";
        const InterfaceInformation& info(infoPair->second.second);
        for(size_t i=0; i < info.size(); i++)
          std::cout<<info[i]<<" ";
        std::cout<<std::endl;
      }
    }
  }

  template<typename R>
  template<typename T1, typename T2>
  inline void Interface<R>::build(const T1& sourceFlags, const T2& destFlags)
  {
    assert(interfaces_.empty());

    // build the seind interface
    InformationBuilder<true> sendInformation(interfaces_);
    this->template buildInterface<R,T1,T2,InformationBuilder<true>,true>(remoteindices_, sourceFlags, destFlags, sendInformation);

    // build the receive interface
    InformationBuilder<false> recvInformation(interfaces_);
    this->template buildInterface<R,T1,T2,InformationBuilder<false>,false>(remoteindices_,sourceFlags, destFlags, recvInformation);
    strip();
  }

  template<typename R>
  inline void Interface<R>::strip()
  {
    typedef InformationMap::iterator const_iterator;
    for(const_iterator interfacePair = interfaces_.begin(); interfacePair != interfaces_.end();)
      if(interfacePair->second.first.size()==0 && interfacePair->second.second.size()==0)
      {
        interfacePair->second.first.free();
        interfacePair->second.second.free();
        const_iterator toerase=interfacePair++;
        interfaces_.erase(toerase);
      }
      else
        ++interfacePair;
  }

  template<typename R>
  inline void Interface<R>::free()
  {
    typedef InformationMap::iterator iterator;
    typedef InformationMap::const_iterator const_iterator;
    const const_iterator end = interfaces_.end();
    for(iterator interfacePair = interfaces_.begin(); interfacePair != end; ++interfacePair)
    {
      interfacePair->second.first.free();
      interfacePair->second.second.free();
    }
    interfaces_.clear();
  }

  template<typename R>
  inline Interface<R>::~Interface()
  {
    free();
  }
  /** @} */
}

namespace std
{
  template<typename R>
  inline ostream& operator<<(ostream& os, const Dune::Interface<R>& interface)
  {
    typedef typename Dune::Interface<R>::InformationMap InfoMap;
    typedef typename InfoMap::const_iterator Iter;
    Iter end = interface.interfaces().end();
    for(Iter i=interface.interfaces().begin(); i!=end; ++i)
    {
      os<<i->first<<": [ source=[";
      for(size_t j=0; j < i->second.first.size(); ++j)
        os<<i->second.first[j]<<" ";
      os<<"] size="<<i->second.first.size()<<", target=[";
      for(size_t j=0; j < i->second.second.size(); ++j)
        os<<i->second.second[j]<<" ";
      os<<"] size="<<i->second.second.size()<<"\n";
    }
    return os;
  }
} // end namespace std

#endif
