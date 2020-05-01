// Generated by gencpp from file grid_map_msgs/ProcessFileResponse.msg
// DO NOT EDIT!


#ifndef GRID_MAP_MSGS_MESSAGE_PROCESSFILERESPONSE_H
#define GRID_MAP_MSGS_MESSAGE_PROCESSFILERESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace grid_map_msgs
{
template <class ContainerAllocator>
struct ProcessFileResponse_
{
  typedef ProcessFileResponse_<ContainerAllocator> Type;

  ProcessFileResponse_()
    : success(false)  {
    }
  ProcessFileResponse_(const ContainerAllocator& _alloc)
    : success(false)  {
  (void)_alloc;
    }



   typedef uint8_t _success_type;
  _success_type success;





  typedef boost::shared_ptr< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> const> ConstPtr;

}; // struct ProcessFileResponse_

typedef ::grid_map_msgs::ProcessFileResponse_<std::allocator<void> > ProcessFileResponse;

typedef boost::shared_ptr< ::grid_map_msgs::ProcessFileResponse > ProcessFileResponsePtr;
typedef boost::shared_ptr< ::grid_map_msgs::ProcessFileResponse const> ProcessFileResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace grid_map_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'grid_map_msgs': ['/tmp/binarydeb/ros-kinetic-grid-map-msgs-1.6.2/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "358e233cde0c8a8bcfea4ce193f8fc15";
  }

  static const char* value(const ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x358e233cde0c8a8bULL;
  static const uint64_t static_value2 = 0xcfea4ce193f8fc15ULL;
};

template<class ContainerAllocator>
struct DataType< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "grid_map_msgs/ProcessFileResponse";
  }

  static const char* value(const ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
\n\
bool success\n\
\n\
";
  }

  static const char* value(const ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.success);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ProcessFileResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::grid_map_msgs::ProcessFileResponse_<ContainerAllocator>& v)
  {
    s << indent << "success: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.success);
  }
};

} // namespace message_operations
} // namespace ros

#endif // GRID_MAP_MSGS_MESSAGE_PROCESSFILERESPONSE_H
