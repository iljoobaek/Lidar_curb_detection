// Generated by gencpp from file jsk_gui_msgs/QueryResponse.msg
// DO NOT EDIT!


#ifndef JSK_GUI_MSGS_MESSAGE_QUERYRESPONSE_H
#define JSK_GUI_MSGS_MESSAGE_QUERYRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace jsk_gui_msgs
{
template <class ContainerAllocator>
struct QueryResponse_
{
  typedef QueryResponse_<ContainerAllocator> Type;

  QueryResponse_()
    : res()  {
    }
  QueryResponse_(const ContainerAllocator& _alloc)
    : res(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _res_type;
  _res_type res;





  typedef boost::shared_ptr< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> const> ConstPtr;

}; // struct QueryResponse_

typedef ::jsk_gui_msgs::QueryResponse_<std::allocator<void> > QueryResponse;

typedef boost::shared_ptr< ::jsk_gui_msgs::QueryResponse > QueryResponsePtr;
typedef boost::shared_ptr< ::jsk_gui_msgs::QueryResponse const> QueryResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace jsk_gui_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'jsk_gui_msgs': ['/tmp/binarydeb/ros-kinetic-jsk-gui-msgs-4.3.1/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "53af918a2a4a2a182c184142fff49b0c";
  }

  static const char* value(const ::jsk_gui_msgs::QueryResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x53af918a2a4a2a18ULL;
  static const uint64_t static_value2 = 0x2c184142fff49b0cULL;
};

template<class ContainerAllocator>
struct DataType< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "jsk_gui_msgs/QueryResponse";
  }

  static const char* value(const ::jsk_gui_msgs::QueryResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string res\n\
";
  }

  static const char* value(const ::jsk_gui_msgs::QueryResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.res);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct QueryResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::jsk_gui_msgs::QueryResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::jsk_gui_msgs::QueryResponse_<ContainerAllocator>& v)
  {
    s << indent << "res: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.res);
  }
};

} // namespace message_operations
} // namespace ros

#endif // JSK_GUI_MSGS_MESSAGE_QUERYRESPONSE_H
