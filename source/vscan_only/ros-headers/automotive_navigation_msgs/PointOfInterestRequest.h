// Generated by gencpp from file automotive_navigation_msgs/PointOfInterestRequest.msg
// DO NOT EDIT!


#ifndef AUTOMOTIVE_NAVIGATION_MSGS_MESSAGE_POINTOFINTERESTREQUEST_H
#define AUTOMOTIVE_NAVIGATION_MSGS_MESSAGE_POINTOFINTERESTREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace automotive_navigation_msgs
{
template <class ContainerAllocator>
struct PointOfInterestRequest_
{
  typedef PointOfInterestRequest_<ContainerAllocator> Type;

  PointOfInterestRequest_()
    : header()
    , name()
    , module_name()
    , request_id(0)
    , cancel(0)
    , update_num(0)
    , guid_valid(0)
    , guid(0)
    , tolerance(0.0)  {
    }
  PointOfInterestRequest_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , name(_alloc)
    , module_name(_alloc)
    , request_id(0)
    , cancel(0)
    , update_num(0)
    , guid_valid(0)
    , guid(0)
    , tolerance(0.0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _name_type;
  _name_type name;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _module_name_type;
  _module_name_type module_name;

   typedef uint16_t _request_id_type;
  _request_id_type request_id;

   typedef uint16_t _cancel_type;
  _cancel_type cancel;

   typedef uint16_t _update_num_type;
  _update_num_type update_num;

   typedef uint16_t _guid_valid_type;
  _guid_valid_type guid_valid;

   typedef uint64_t _guid_type;
  _guid_type guid;

   typedef float _tolerance_type;
  _tolerance_type tolerance;





  typedef boost::shared_ptr< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> const> ConstPtr;

}; // struct PointOfInterestRequest_

typedef ::automotive_navigation_msgs::PointOfInterestRequest_<std::allocator<void> > PointOfInterestRequest;

typedef boost::shared_ptr< ::automotive_navigation_msgs::PointOfInterestRequest > PointOfInterestRequestPtr;
typedef boost::shared_ptr< ::automotive_navigation_msgs::PointOfInterestRequest const> PointOfInterestRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace automotive_navigation_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'automotive_navigation_msgs': ['/tmp/binarydeb/ros-kinetic-automotive-navigation-msgs-3.0.3/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "b55c53f232d0288e56995c132cf04930";
  }

  static const char* value(const ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xb55c53f232d0288eULL;
  static const uint64_t static_value2 = 0x56995c132cf04930ULL;
};

template<class ContainerAllocator>
struct DataType< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "automotive_navigation_msgs/PointOfInterestRequest";
  }

  static const char* value(const ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Point of Interest Request Message\n\
# Contains information needed to request point of interest information\n\
\n\
std_msgs/Header header\n\
\n\
string name        # Name of the point of interest list\n\
\n\
string module_name # module name of the requesting node\n\
\n\
uint16 request_id  # Unique id of this request\n\
                   # Can make another request with the same requestId and\n\
                   # different update_num, guid, or tolerance.  New one will\n\
                   # replace the old one.\n\
\n\
uint16 cancel      # Set to 1 to cancel the request with this requestId\n\
\n\
uint16 update_num  # The update number of the point list to use\n\
\n\
uint16 guid_valid  # Request is for a specific point, not all points in list\n\
uint64 guid        # The unique Id for the desired point\n\
\n\
float32 tolerance  # How close to the current vehicle's position a point needs to be\n\
\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
";
  }

  static const char* value(const ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.name);
      stream.next(m.module_name);
      stream.next(m.request_id);
      stream.next(m.cancel);
      stream.next(m.update_num);
      stream.next(m.guid_valid);
      stream.next(m.guid);
      stream.next(m.tolerance);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PointOfInterestRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::automotive_navigation_msgs::PointOfInterestRequest_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.name);
    s << indent << "module_name: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.module_name);
    s << indent << "request_id: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.request_id);
    s << indent << "cancel: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.cancel);
    s << indent << "update_num: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.update_num);
    s << indent << "guid_valid: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.guid_valid);
    s << indent << "guid: ";
    Printer<uint64_t>::stream(s, indent + "  ", v.guid);
    s << indent << "tolerance: ";
    Printer<float>::stream(s, indent + "  ", v.tolerance);
  }
};

} // namespace message_operations
} // namespace ros

#endif // AUTOMOTIVE_NAVIGATION_MSGS_MESSAGE_POINTOFINTERESTREQUEST_H
