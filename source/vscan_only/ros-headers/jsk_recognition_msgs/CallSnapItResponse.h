// Generated by gencpp from file jsk_recognition_msgs/CallSnapItResponse.msg
// DO NOT EDIT!


#ifndef JSK_RECOGNITION_MSGS_MESSAGE_CALLSNAPITRESPONSE_H
#define JSK_RECOGNITION_MSGS_MESSAGE_CALLSNAPITRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/Pose.h>

namespace jsk_recognition_msgs
{
template <class ContainerAllocator>
struct CallSnapItResponse_
{
  typedef CallSnapItResponse_<ContainerAllocator> Type;

  CallSnapItResponse_()
    : transformation()  {
    }
  CallSnapItResponse_(const ContainerAllocator& _alloc)
    : transformation(_alloc)  {
  (void)_alloc;
    }



   typedef  ::geometry_msgs::Pose_<ContainerAllocator>  _transformation_type;
  _transformation_type transformation;





  typedef boost::shared_ptr< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> const> ConstPtr;

}; // struct CallSnapItResponse_

typedef ::jsk_recognition_msgs::CallSnapItResponse_<std::allocator<void> > CallSnapItResponse;

typedef boost::shared_ptr< ::jsk_recognition_msgs::CallSnapItResponse > CallSnapItResponsePtr;
typedef boost::shared_ptr< ::jsk_recognition_msgs::CallSnapItResponse const> CallSnapItResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace jsk_recognition_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'pcl_msgs': ['/opt/ros/kinetic/share/pcl_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/kinetic/share/sensor_msgs/cmake/../msg'], 'jsk_footstep_msgs': ['/opt/ros/kinetic/share/jsk_footstep_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'jsk_recognition_msgs': ['/tmp/binarydeb/ros-kinetic-jsk-recognition-msgs-1.2.9/msg'], 'actionlib_msgs': ['/opt/ros/kinetic/share/actionlib_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d787e2767b5ea7b19a81c647df92a8de";
  }

  static const char* value(const ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd787e2767b5ea7b1ULL;
  static const uint64_t static_value2 = 0x9a81c647df92a8deULL;
};

template<class ContainerAllocator>
struct DataType< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "jsk_recognition_msgs/CallSnapItResponse";
  }

  static const char* value(const ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "geometry_msgs/Pose transformation\n\
\n\
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

  static const char* value(const ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.transformation);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct CallSnapItResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::jsk_recognition_msgs::CallSnapItResponse_<ContainerAllocator>& v)
  {
    s << indent << "transformation: ";
    s << std::endl;
    Printer< ::geometry_msgs::Pose_<ContainerAllocator> >::stream(s, indent + "  ", v.transformation);
  }
};

} // namespace message_operations
} // namespace ros

#endif // JSK_RECOGNITION_MSGS_MESSAGE_CALLSNAPITRESPONSE_H
