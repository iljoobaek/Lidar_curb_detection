// Generated by gencpp from file jsk_gui_msgs/YesNoRequest.msg
// DO NOT EDIT!


#ifndef JSK_GUI_MSGS_MESSAGE_YESNOREQUEST_H
#define JSK_GUI_MSGS_MESSAGE_YESNOREQUEST_H


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
struct YesNoRequest_
{
  typedef YesNoRequest_<ContainerAllocator> Type;

  YesNoRequest_()
    : message()  {
    }
  YesNoRequest_(const ContainerAllocator& _alloc)
    : message(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _message_type;
  _message_type message;





  typedef boost::shared_ptr< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> const> ConstPtr;

}; // struct YesNoRequest_

typedef ::jsk_gui_msgs::YesNoRequest_<std::allocator<void> > YesNoRequest;

typedef boost::shared_ptr< ::jsk_gui_msgs::YesNoRequest > YesNoRequestPtr;
typedef boost::shared_ptr< ::jsk_gui_msgs::YesNoRequest const> YesNoRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5f003d6bcc824cbd51361d66d8e4f76c";
  }

  static const char* value(const ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5f003d6bcc824cbdULL;
  static const uint64_t static_value2 = 0x51361d66d8e4f76cULL;
};

template<class ContainerAllocator>
struct DataType< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "jsk_gui_msgs/YesNoRequest";
  }

  static const char* value(const ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string message\n\
";
  }

  static const char* value(const ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.message);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct YesNoRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::jsk_gui_msgs::YesNoRequest_<ContainerAllocator>& v)
  {
    s << indent << "message: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.message);
  }
};

} // namespace message_operations
} // namespace ros

#endif // JSK_GUI_MSGS_MESSAGE_YESNOREQUEST_H
