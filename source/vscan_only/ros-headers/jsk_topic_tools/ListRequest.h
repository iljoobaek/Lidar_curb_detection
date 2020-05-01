// Generated by gencpp from file jsk_topic_tools/ListRequest.msg
// DO NOT EDIT!


#ifndef JSK_TOPIC_TOOLS_MESSAGE_LISTREQUEST_H
#define JSK_TOPIC_TOOLS_MESSAGE_LISTREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace jsk_topic_tools
{
template <class ContainerAllocator>
struct ListRequest_
{
  typedef ListRequest_<ContainerAllocator> Type;

  ListRequest_()
    {
    }
  ListRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::jsk_topic_tools::ListRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::jsk_topic_tools::ListRequest_<ContainerAllocator> const> ConstPtr;

}; // struct ListRequest_

typedef ::jsk_topic_tools::ListRequest_<std::allocator<void> > ListRequest;

typedef boost::shared_ptr< ::jsk_topic_tools::ListRequest > ListRequestPtr;
typedef boost::shared_ptr< ::jsk_topic_tools::ListRequest const> ListRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::jsk_topic_tools::ListRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace jsk_topic_tools

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'jsk_topic_tools': ['/tmp/binarydeb/ros-kinetic-jsk-topic-tools-2.2.10/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::jsk_topic_tools::ListRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_topic_tools::ListRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_topic_tools::ListRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::jsk_topic_tools::ListRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "jsk_topic_tools/ListRequest";
  }

  static const char* value(const ::jsk_topic_tools::ListRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
";
  }

  static const char* value(const ::jsk_topic_tools::ListRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ListRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::jsk_topic_tools::ListRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::jsk_topic_tools::ListRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // JSK_TOPIC_TOOLS_MESSAGE_LISTREQUEST_H
