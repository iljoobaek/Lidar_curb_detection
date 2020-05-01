// Generated by gencpp from file teleop_tools_msgs/IncrementGoal.msg
// DO NOT EDIT!


#ifndef TELEOP_TOOLS_MSGS_MESSAGE_INCREMENTGOAL_H
#define TELEOP_TOOLS_MSGS_MESSAGE_INCREMENTGOAL_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace teleop_tools_msgs
{
template <class ContainerAllocator>
struct IncrementGoal_
{
  typedef IncrementGoal_<ContainerAllocator> Type;

  IncrementGoal_()
    : increment_by()  {
    }
  IncrementGoal_(const ContainerAllocator& _alloc)
    : increment_by(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _increment_by_type;
  _increment_by_type increment_by;





  typedef boost::shared_ptr< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> const> ConstPtr;

}; // struct IncrementGoal_

typedef ::teleop_tools_msgs::IncrementGoal_<std::allocator<void> > IncrementGoal;

typedef boost::shared_ptr< ::teleop_tools_msgs::IncrementGoal > IncrementGoalPtr;
typedef boost::shared_ptr< ::teleop_tools_msgs::IncrementGoal const> IncrementGoalConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace teleop_tools_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'teleop_tools_msgs': ['/tmp/binarydeb/ros-kinetic-teleop-tools-msgs-0.3.1/obj-x86_64-linux-gnu/devel/share/teleop_tools_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
{
  static const char* value()
  {
    return "6801194f49dae2bee5b8f17b6ce4596f";
  }

  static const char* value(const ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x6801194f49dae2beULL;
  static const uint64_t static_value2 = 0xe5b8f17b6ce4596fULL;
};

template<class ContainerAllocator>
struct DataType< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
{
  static const char* value()
  {
    return "teleop_tools_msgs/IncrementGoal";
  }

  static const char* value(const ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n\
float32[] increment_by\n\
";
  }

  static const char* value(const ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.increment_by);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct IncrementGoal_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::teleop_tools_msgs::IncrementGoal_<ContainerAllocator>& v)
  {
    s << indent << "increment_by[]" << std::endl;
    for (size_t i = 0; i < v.increment_by.size(); ++i)
    {
      s << indent << "  increment_by[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.increment_by[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // TELEOP_TOOLS_MSGS_MESSAGE_INCREMENTGOAL_H
