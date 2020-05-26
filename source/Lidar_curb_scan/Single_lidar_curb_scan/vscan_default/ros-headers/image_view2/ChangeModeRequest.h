// Generated by gencpp from file image_view2/ChangeModeRequest.msg
// DO NOT EDIT!


#ifndef IMAGE_VIEW2_MESSAGE_CHANGEMODEREQUEST_H
#define IMAGE_VIEW2_MESSAGE_CHANGEMODEREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace image_view2
{
template <class ContainerAllocator>
struct ChangeModeRequest_
{
  typedef ChangeModeRequest_<ContainerAllocator> Type;

  ChangeModeRequest_()
    : mode()  {
    }
  ChangeModeRequest_(const ContainerAllocator& _alloc)
    : mode(_alloc)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _mode_type;
  _mode_type mode;





  typedef boost::shared_ptr< ::image_view2::ChangeModeRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::image_view2::ChangeModeRequest_<ContainerAllocator> const> ConstPtr;

}; // struct ChangeModeRequest_

typedef ::image_view2::ChangeModeRequest_<std::allocator<void> > ChangeModeRequest;

typedef boost::shared_ptr< ::image_view2::ChangeModeRequest > ChangeModeRequestPtr;
typedef boost::shared_ptr< ::image_view2::ChangeModeRequest const> ChangeModeRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::image_view2::ChangeModeRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::image_view2::ChangeModeRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace image_view2

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'image_view2': ['/tmp/binarydeb/ros-kinetic-image-view2-2.2.10/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::image_view2::ChangeModeRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::image_view2::ChangeModeRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::image_view2::ChangeModeRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "e84dc3ad5dc323bb64f0aca01c2d1eef";
  }

  static const char* value(const ::image_view2::ChangeModeRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xe84dc3ad5dc323bbULL;
  static const uint64_t static_value2 = 0x64f0aca01c2d1eefULL;
};

template<class ContainerAllocator>
struct DataType< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "image_view2/ChangeModeRequest";
  }

  static const char* value(const ::image_view2::ChangeModeRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string mode\n\
";
  }

  static const char* value(const ::image_view2::ChangeModeRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.mode);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct ChangeModeRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::image_view2::ChangeModeRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::image_view2::ChangeModeRequest_<ContainerAllocator>& v)
  {
    s << indent << "mode: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.mode);
  }
};

} // namespace message_operations
} // namespace ros

#endif // IMAGE_VIEW2_MESSAGE_CHANGEMODEREQUEST_H