// Generated by gencpp from file pacmod_msgs/DoorRpt.msg
// DO NOT EDIT!


#ifndef PACMOD_MSGS_MESSAGE_DOORRPT_H
#define PACMOD_MSGS_MESSAGE_DOORRPT_H


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
struct DoorRpt_
{
  typedef DoorRpt_<ContainerAllocator> Type;

  DoorRpt_()
    : header()
    , driver_door_open(false)
    , driver_door_open_is_valid(false)
    , passenger_door_open(false)
    , passenger_door_open_is_valid(false)
    , rear_driver_door_open(false)
    , rear_driver_door_open_is_valid(false)
    , rear_passenger_door_open(false)
    , rear_passenger_door_open_is_valid(false)
    , hood_open(false)
    , hood_open_is_valid(false)
    , trunk_open(false)
    , trunk_open_is_valid(false)
    , fuel_door_open(false)
    , fuel_door_open_is_valid(false)  {
    }
  DoorRpt_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , driver_door_open(false)
    , driver_door_open_is_valid(false)
    , passenger_door_open(false)
    , passenger_door_open_is_valid(false)
    , rear_driver_door_open(false)
    , rear_driver_door_open_is_valid(false)
    , rear_passenger_door_open(false)
    , rear_passenger_door_open_is_valid(false)
    , hood_open(false)
    , hood_open_is_valid(false)
    , trunk_open(false)
    , trunk_open_is_valid(false)
    , fuel_door_open(false)
    , fuel_door_open_is_valid(false)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef uint8_t _driver_door_open_type;
  _driver_door_open_type driver_door_open;

   typedef uint8_t _driver_door_open_is_valid_type;
  _driver_door_open_is_valid_type driver_door_open_is_valid;

   typedef uint8_t _passenger_door_open_type;
  _passenger_door_open_type passenger_door_open;

   typedef uint8_t _passenger_door_open_is_valid_type;
  _passenger_door_open_is_valid_type passenger_door_open_is_valid;

   typedef uint8_t _rear_driver_door_open_type;
  _rear_driver_door_open_type rear_driver_door_open;

   typedef uint8_t _rear_driver_door_open_is_valid_type;
  _rear_driver_door_open_is_valid_type rear_driver_door_open_is_valid;

   typedef uint8_t _rear_passenger_door_open_type;
  _rear_passenger_door_open_type rear_passenger_door_open;

   typedef uint8_t _rear_passenger_door_open_is_valid_type;
  _rear_passenger_door_open_is_valid_type rear_passenger_door_open_is_valid;

   typedef uint8_t _hood_open_type;
  _hood_open_type hood_open;

   typedef uint8_t _hood_open_is_valid_type;
  _hood_open_is_valid_type hood_open_is_valid;

   typedef uint8_t _trunk_open_type;
  _trunk_open_type trunk_open;

   typedef uint8_t _trunk_open_is_valid_type;
  _trunk_open_is_valid_type trunk_open_is_valid;

   typedef uint8_t _fuel_door_open_type;
  _fuel_door_open_type fuel_door_open;

   typedef uint8_t _fuel_door_open_is_valid_type;
  _fuel_door_open_is_valid_type fuel_door_open_is_valid;





  typedef boost::shared_ptr< ::pacmod_msgs::DoorRpt_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pacmod_msgs::DoorRpt_<ContainerAllocator> const> ConstPtr;

}; // struct DoorRpt_

typedef ::pacmod_msgs::DoorRpt_<std::allocator<void> > DoorRpt;

typedef boost::shared_ptr< ::pacmod_msgs::DoorRpt > DoorRptPtr;
typedef boost::shared_ptr< ::pacmod_msgs::DoorRpt const> DoorRptConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pacmod_msgs::DoorRpt_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pacmod_msgs::DoorRpt_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pacmod_msgs::DoorRpt_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pacmod_msgs::DoorRpt_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a2ffa235d04f8d5d5e349a5d9caead12";
  }

  static const char* value(const ::pacmod_msgs::DoorRpt_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa2ffa235d04f8d5dULL;
  static const uint64_t static_value2 = 0x5e349a5d9caead12ULL;
};

template<class ContainerAllocator>
struct DataType< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pacmod_msgs/DoorRpt";
  }

  static const char* value(const ::pacmod_msgs::DoorRpt_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n\
\n\
bool driver_door_open\n\
bool driver_door_open_is_valid\n\
bool passenger_door_open\n\
bool passenger_door_open_is_valid\n\
bool rear_driver_door_open\n\
bool rear_driver_door_open_is_valid\n\
bool rear_passenger_door_open\n\
bool rear_passenger_door_open_is_valid\n\
bool hood_open\n\
bool hood_open_is_valid\n\
bool trunk_open\n\
bool trunk_open_is_valid\n\
bool fuel_door_open\n\
bool fuel_door_open_is_valid\n\
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

  static const char* value(const ::pacmod_msgs::DoorRpt_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.driver_door_open);
      stream.next(m.driver_door_open_is_valid);
      stream.next(m.passenger_door_open);
      stream.next(m.passenger_door_open_is_valid);
      stream.next(m.rear_driver_door_open);
      stream.next(m.rear_driver_door_open_is_valid);
      stream.next(m.rear_passenger_door_open);
      stream.next(m.rear_passenger_door_open_is_valid);
      stream.next(m.hood_open);
      stream.next(m.hood_open_is_valid);
      stream.next(m.trunk_open);
      stream.next(m.trunk_open_is_valid);
      stream.next(m.fuel_door_open);
      stream.next(m.fuel_door_open_is_valid);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct DoorRpt_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pacmod_msgs::DoorRpt_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pacmod_msgs::DoorRpt_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "driver_door_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.driver_door_open);
    s << indent << "driver_door_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.driver_door_open_is_valid);
    s << indent << "passenger_door_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.passenger_door_open);
    s << indent << "passenger_door_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.passenger_door_open_is_valid);
    s << indent << "rear_driver_door_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.rear_driver_door_open);
    s << indent << "rear_driver_door_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.rear_driver_door_open_is_valid);
    s << indent << "rear_passenger_door_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.rear_passenger_door_open);
    s << indent << "rear_passenger_door_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.rear_passenger_door_open_is_valid);
    s << indent << "hood_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.hood_open);
    s << indent << "hood_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.hood_open_is_valid);
    s << indent << "trunk_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.trunk_open);
    s << indent << "trunk_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.trunk_open_is_valid);
    s << indent << "fuel_door_open: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.fuel_door_open);
    s << indent << "fuel_door_open_is_valid: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.fuel_door_open_is_valid);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PACMOD_MSGS_MESSAGE_DOORRPT_H