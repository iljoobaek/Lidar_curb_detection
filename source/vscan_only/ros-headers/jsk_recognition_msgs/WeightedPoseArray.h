// Generated by gencpp from file jsk_recognition_msgs/WeightedPoseArray.msg
// DO NOT EDIT!


#ifndef JSK_RECOGNITION_MSGS_MESSAGE_WEIGHTEDPOSEARRAY_H
#define JSK_RECOGNITION_MSGS_MESSAGE_WEIGHTEDPOSEARRAY_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/PoseArray.h>

namespace jsk_recognition_msgs
{
template <class ContainerAllocator>
struct WeightedPoseArray_
{
  typedef WeightedPoseArray_<ContainerAllocator> Type;

  WeightedPoseArray_()
    : header()
    , weights()
    , poses()  {
    }
  WeightedPoseArray_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , weights(_alloc)
    , poses(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _weights_type;
  _weights_type weights;

   typedef  ::geometry_msgs::PoseArray_<ContainerAllocator>  _poses_type;
  _poses_type poses;





  typedef boost::shared_ptr< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> const> ConstPtr;

}; // struct WeightedPoseArray_

typedef ::jsk_recognition_msgs::WeightedPoseArray_<std::allocator<void> > WeightedPoseArray;

typedef boost::shared_ptr< ::jsk_recognition_msgs::WeightedPoseArray > WeightedPoseArrayPtr;
typedef boost::shared_ptr< ::jsk_recognition_msgs::WeightedPoseArray const> WeightedPoseArrayConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace jsk_recognition_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'pcl_msgs': ['/opt/ros/kinetic/share/pcl_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'jsk_footstep_msgs': ['/opt/ros/kinetic/share/jsk_footstep_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'jsk_recognition_msgs': ['/tmp/binarydeb/ros-kinetic-jsk-recognition-msgs-1.2.9/msg'], 'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "40f180494a75a8797b1c2ba81b2cb4c0";
  }

  static const char* value(const ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x40f180494a75a879ULL;
  static const uint64_t static_value2 = 0x7b1c2ba81b2cb4c0ULL;
};

template<class ContainerAllocator>
struct DataType< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "jsk_recognition_msgs/WeightedPoseArray";
  }

  static const char* value(const ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
float32[] weights\n\
geometry_msgs/PoseArray poses\n\
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
\n\
================================================================================\n\
MSG: geometry_msgs/PoseArray\n\
# An array of poses with a header for global reference.\n\
\n\
Header header\n\
\n\
Pose[] poses\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Pose\n\
# A representation of pose in free space, composed of position and orientation. \n\
Point position\n\
Quaternion orientation\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
# This contains the position of a point in free space\n\
float64 x\n\
float64 y\n\
float64 z\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Quaternion\n\
# This represents an orientation in free space in quaternion form.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
float64 w\n\
";
  }

  static const char* value(const ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.weights);
      stream.next(m.poses);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct WeightedPoseArray_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::jsk_recognition_msgs::WeightedPoseArray_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "weights[]" << std::endl;
    for (size_t i = 0; i < v.weights.size(); ++i)
    {
      s << indent << "  weights[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.weights[i]);
    }
    s << indent << "poses: ";
    s << std::endl;
    Printer< ::geometry_msgs::PoseArray_<ContainerAllocator> >::stream(s, indent + "  ", v.poses);
  }
};

} // namespace message_operations
} // namespace ros

#endif // JSK_RECOGNITION_MSGS_MESSAGE_WEIGHTEDPOSEARRAY_H
