// Generated by gencpp from file pacmod_msgs/AccelAuxRpt.msg
// DO NOT EDIT!


#ifndef PACMOD_MSGS_MESSAGE_ACCELAUXRPT_H
#define PACMOD_MSGS_MESSAGE_ACCELAUXRPT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>

namespace pacmod_msgs
{
template <class ContainerAllocator>
struct AccelAuxRpt_
{
  typedef AccelAuxRpt_<ContainerAllocator> Type;

  AccelAuxRpt_()
    : header()
    , raw_pedal_pos(0.0)
    , raw_pedal_pos_is_valid(false)
    , raw_pedal_force(0.0)
    , raw_pedal_force_is_valid(false)
    , user_interaction(false)
    , user_interaction_is_valid(false)
    , brake_interlock_active(false)
    , brake_interlock_active_is_valid(false)  {
    }
  AccelAuxRpt_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , raw_pedal_pos(0.0)
    , raw_pedal_pos_is_valid(false)
    , raw_pedal_force(0.0)
    , raw_pedal_force_is_valid(false)
    , user_interaction(false)
    , user_interaction_is_valid(false)
    , brake_interlock_active(false)
    , brake_interlock_active_is_valid(false)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef double _raw_pedal_pos_type;
  _raw_pedal_pos_type raw_pedal_pos;

   typedef uint8_t _raw_pedal_pos_is_valid_type;
  _raw_pedal_pos_is_valid_type raw_pedal_pos_is_valid;

   typedef double _raw_pedal_force_type;
  _raw_pedal_force_type raw_pedal_force;

   typedef uint8_t _raw_pedal_force_is_valid_type;
  _raw_pedal_force_is_valid_type raw_pedal_force_is_valid;

   typedef uint8_t _user_interaction_type;
  _user_interaction_type user_interaction;

   typedef uint8_t _user_interaction_is_valid_type;
  _user_interaction_is_valid_type user_interaction_is_valid;

   typedef uint8_t _brake_interlock_active_type;
  _brake_interlock_active_type brake_interlock_active;

   typedef uint8_t _brake_interlock_active_is_valid_type;
  _brake_interlock_active_is_valid_type brake_interlock_active_is_valid;





  typedef boost::shared_ptr< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> const> ConstPtr;

}; // struct AccelAuxRpt_

typedef ::pacmod_msgs::AccelAuxRpt_<std::allocator<void> > AccelAuxRpt;

typedef boost::shared_ptr< ::pacmod_msgs::AccelAuxRpt > AccelAuxRptPtr;
typedef boost::shared_ptr< ::pacmod_msgs::AccelAuxRpt const> AccelAuxRptConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace pacmod_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'pacmod_msgs': ['/tmp/binarydeb/ros-kinetic-pacmod-msgs-3.0.1/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
{
  static const char* value()
  {
    return "2f644f02020323fdb0afab1a11b54b70";
  }

  static const char* value(const ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x2f644f02020323fdULL;
  static const uint64_t static_value2 = 0xb0afab1a11b54b70ULL;
};

template<class ContainerAllocator>
struct DataType< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pacmod_msgs/AccelAuxRpt";
  }

  static const char* value(const ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n\
\n\
float64 raw_pedal_pos\n\
bool raw_pedal_pos_is_valid\n\
float64 raw_pedal_force\n\
bool raw_pedal_force_is_valid\n\
bool user_interaction\n\
bool user_interaction_is_valid\n\
bool brake_interlock_active\n\
bool brake_interlock_active_is_valid\n\
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

  static const char* value(const ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.raw_pedal_pos);
      stream.next(m.raw_pedal_pos_is_valid);
      stream.next(m.raw_pedal_force);
      stream.next(m.raw_pedal_force_is_valid);
      stream.next(m.user_interaction);
      stream.next(m.user_interaction_is_valid);
      stream.next(m.brake_interlock_active);
      stream.next(m.brake_interlock_active_is_valid);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct AccelAuxRpt_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pacmod_msgs::AccelAuxRpt_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "raw_pedal_pos: ";
    Printer<double>::stream(s, indent + "  ", v.raw_pedal_pos);
    s << indent << "raw_pedal_pos_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.raw_pedal_pos_is_valid);
    s << indent << "raw_pedal_force: ";
    Printer<double>::stream(s, indent + "  ", v.raw_pedal_force);
    s << indent << "raw_pedal_force_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.raw_pedal_force_is_valid);
    s << indent << "user_interaction: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.user_interaction);
    s << indent << "user_interaction_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.user_interaction_is_valid);
    s << indent << "brake_interlock_active: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.brake_interlock_active);
    s << indent << "brake_interlock_active_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.brake_interlock_active_is_valid);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PACMOD_MSGS_MESSAGE_ACCELAUXRPT_H
