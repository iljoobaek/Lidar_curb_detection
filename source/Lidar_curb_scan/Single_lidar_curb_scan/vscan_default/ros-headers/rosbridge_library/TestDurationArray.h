// Generated by gencpp from file rosbridge_library/TestDurationArray.msg
// DO NOT EDIT!


#ifndef ROSBRIDGE_LIBRARY_MESSAGE_TESTDURATIONARRAY_H
#define ROSBRIDGE_LIBRARY_MESSAGE_TESTDURATIONARRAY_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace rosbridge_library
{
template <class ContainerAllocator>
struct TestDurationArray_
{
  typedef TestDurationArray_<ContainerAllocator> Type;

  TestDurationArray_()
    : durations()  {
    }
  TestDurationArray_(const ContainerAllocator& _alloc)
    : durations(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<ros::Duration, typename ContainerAllocator::template rebind<ros::Duration>::other >  _durations_type;
  _durations_type durations;





  typedef boost::shared_ptr< ::rosbridge_library::TestDurationArray_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::rosbridge_library::TestDurationArray_<ContainerAllocator> const> ConstPtr;

}; // struct TestDurationArray_

typedef ::rosbridge_library::TestDurationArray_<std::allocator<void> > TestDurationArray;

typedef boost::shared_ptr< ::rosbridge_library::TestDurationArray > TestDurationArrayPtr;
typedef boost::shared_ptr< ::rosbridge_library::TestDurationArray const> TestDurationArrayConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::rosbridge_library::TestDurationArray_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace rosbridge_library

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'rosbridge_library': ['/tmp/binarydeb/ros-kinetic-rosbridge-library-0.11.4/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::rosbridge_library::TestDurationArray_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::rosbridge_library::TestDurationArray_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::rosbridge_library::TestDurationArray_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "8b3bcadc803a7fcbc857c6a1dab53bcd";
  }

  static const char* value(const ::rosbridge_library::TestDurationArray_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x8b3bcadc803a7fcbULL;
  static const uint64_t static_value2 = 0xc857c6a1dab53bcdULL;
};

template<class ContainerAllocator>
struct DataType< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "rosbridge_library/TestDurationArray";
  }

  static const char* value(const ::rosbridge_library::TestDurationArray_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "duration[] durations\n\
";
  }

  static const char* value(const ::rosbridge_library::TestDurationArray_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.durations);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TestDurationArray_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::rosbridge_library::TestDurationArray_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::rosbridge_library::TestDurationArray_<ContainerAllocator>& v)
  {
    s << indent << "durations[]" << std::endl;
    for (size_t i = 0; i < v.durations.size(); ++i)
    {
      s << indent << "  durations[" << i << "]: ";
      Printer<ros::Duration>::stream(s, indent + "  ", v.durations[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // ROSBRIDGE_LIBRARY_MESSAGE_TESTDURATIONARRAY_H