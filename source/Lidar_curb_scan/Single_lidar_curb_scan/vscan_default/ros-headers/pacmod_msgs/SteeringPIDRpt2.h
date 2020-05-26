// Generated by gencpp from file pacmod_msgs/SteeringPIDRpt2.msg
// DO NOT EDIT!


#ifndef PACMOD_MSGS_MESSAGE_STEERINGPIDRPT2_H
#define PACMOD_MSGS_MESSAGE_STEERINGPIDRPT2_H


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
struct SteeringPIDRpt2_
{
  typedef SteeringPIDRpt2_<ContainerAllocator> Type;

  SteeringPIDRpt2_()
    : header()
    , p_term(0.0)
    , i_term(0.0)
    , d_term(0.0)
    , all_terms(0.0)  {
    }
  SteeringPIDRpt2_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , p_term(0.0)
    , i_term(0.0)
    , d_term(0.0)
    , all_terms(0.0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef double _p_term_type;
  _p_term_type p_term;

   typedef double _i_term_type;
  _i_term_type i_term;

   typedef double _d_term_type;
  _d_term_type d_term;

   typedef double _all_terms_type;
  _all_terms_type all_terms;





  typedef boost::shared_ptr< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> const> ConstPtr;

}; // struct SteeringPIDRpt2_

typedef ::pacmod_msgs::SteeringPIDRpt2_<std::allocator<void> > SteeringPIDRpt2;

typedef boost::shared_ptr< ::pacmod_msgs::SteeringPIDRpt2 > SteeringPIDRpt2Ptr;
typedef boost::shared_ptr< ::pacmod_msgs::SteeringPIDRpt2 const> SteeringPIDRpt2ConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >::stream(s, "", v);
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
struct IsFixedSize< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
{
  static const char* value()
  {
    return "1adfcb7e7b84f38f1763878f5d8e8ff5";
  }

  static const char* value(const ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x1adfcb7e7b84f38fULL;
  static const uint64_t static_value2 = 0x1763878f5d8e8ff5ULL;
};

template<class ContainerAllocator>
struct DataType< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
{
  static const char* value()
  {
    return "pacmod_msgs/SteeringPIDRpt2";
  }

  static const char* value(const ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n\
\n\
float64 p_term\n\
float64 i_term\n\
float64 d_term\n\
float64 all_terms      # sum of P, I, and D terms\n\
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

  static const char* value(const ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.p_term);
      stream.next(m.i_term);
      stream.next(m.d_term);
      stream.next(m.all_terms);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct SteeringPIDRpt2_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::pacmod_msgs::SteeringPIDRpt2_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "p_term: ";
    Printer<double>::stream(s, indent + "  ", v.p_term);
    s << indent << "i_term: ";
    Printer<double>::stream(s, indent + "  ", v.i_term);
    s << indent << "d_term: ";
    Printer<double>::stream(s, indent + "  ", v.d_term);
    s << indent << "all_terms: ";
    Printer<double>::stream(s, indent + "  ", v.all_terms);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PACMOD_MSGS_MESSAGE_STEERINGPIDRPT2_H