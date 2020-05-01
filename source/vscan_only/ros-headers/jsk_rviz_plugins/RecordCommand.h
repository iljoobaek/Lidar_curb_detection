// Generated by gencpp from file jsk_rviz_plugins/RecordCommand.msg
// DO NOT EDIT!


#ifndef JSK_RVIZ_PLUGINS_MESSAGE_RECORDCOMMAND_H
#define JSK_RVIZ_PLUGINS_MESSAGE_RECORDCOMMAND_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace jsk_rviz_plugins
{
template <class ContainerAllocator>
struct RecordCommand_
{
  typedef RecordCommand_<ContainerAllocator> Type;

  RecordCommand_()
    : command(0)
    , target()  {
    }
  RecordCommand_(const ContainerAllocator& _alloc)
    : command(0)
    , target(_alloc)  {
  (void)_alloc;
    }



   typedef int8_t _command_type;
  _command_type command;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _target_type;
  _target_type target;



  enum {
    RECORD = 0u,
    RECORD_STOP = 1u,
    PLAY = 2u,
  };


  typedef boost::shared_ptr< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> const> ConstPtr;

}; // struct RecordCommand_

typedef ::jsk_rviz_plugins::RecordCommand_<std::allocator<void> > RecordCommand;

typedef boost::shared_ptr< ::jsk_rviz_plugins::RecordCommand > RecordCommandPtr;
typedef boost::shared_ptr< ::jsk_rviz_plugins::RecordCommand const> RecordCommandConstPtr;

// constants requiring out of line definition

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace jsk_rviz_plugins

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'jsk_rviz_plugins': ['/tmp/binarydeb/ros-kinetic-jsk-rviz-plugins-2.1.5/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "31931c62eab5500089183eef0161c139";
  }

  static const char* value(const ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x31931c62eab55000ULL;
  static const uint64_t static_value2 = 0x89183eef0161c139ULL;
};

template<class ContainerAllocator>
struct DataType< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "jsk_rviz_plugins/RecordCommand";
  }

  static const char* value(const ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 RECORD=0\n\
uint8 RECORD_STOP=1\n\
uint8 PLAY=2\n\
\n\
int8 command\n\
string target\n\
";
  }

  static const char* value(const ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.command);
      stream.next(m.target);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RecordCommand_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::jsk_rviz_plugins::RecordCommand_<ContainerAllocator>& v)
  {
    s << indent << "command: ";
    Printer<int8_t>::stream(s, indent + "  ", v.command);
    s << indent << "target: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.target);
  }
};

} // namespace message_operations
} // namespace ros

#endif // JSK_RVIZ_PLUGINS_MESSAGE_RECORDCOMMAND_H
