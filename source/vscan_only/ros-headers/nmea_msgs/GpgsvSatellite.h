// Generated by gencpp from file nmea_msgs/GpgsvSatellite.msg
// DO NOT EDIT!


#ifndef NMEA_MSGS_MESSAGE_GPGSVSATELLITE_H
#define NMEA_MSGS_MESSAGE_GPGSVSATELLITE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace nmea_msgs
{
template <class ContainerAllocator>
struct GpgsvSatellite_
{
  typedef GpgsvSatellite_<ContainerAllocator> Type;

  GpgsvSatellite_()
    : prn(0)
    , elevation(0)
    , azimuth(0)
    , snr(0)  {
    }
  GpgsvSatellite_(const ContainerAllocator& _alloc)
    : prn(0)
    , elevation(0)
    , azimuth(0)
    , snr(0)  {
  (void)_alloc;
    }



   typedef uint8_t _prn_type;
  _prn_type prn;

   typedef uint8_t _elevation_type;
  _elevation_type elevation;

   typedef uint16_t _azimuth_type;
  _azimuth_type azimuth;

   typedef int8_t _snr_type;
  _snr_type snr;





  typedef boost::shared_ptr< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> const> ConstPtr;

}; // struct GpgsvSatellite_

typedef ::nmea_msgs::GpgsvSatellite_<std::allocator<void> > GpgsvSatellite;

typedef boost::shared_ptr< ::nmea_msgs::GpgsvSatellite > GpgsvSatellitePtr;
typedef boost::shared_ptr< ::nmea_msgs::GpgsvSatellite const> GpgsvSatelliteConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace nmea_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'nmea_msgs': ['/tmp/binarydeb/ros-kinetic-nmea-msgs-1.1.0/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d862f2ce05a26a83264a8add99c7b668";
  }

  static const char* value(const ::nmea_msgs::GpgsvSatellite_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd862f2ce05a26a83ULL;
  static const uint64_t static_value2 = 0x264a8add99c7b668ULL;
};

template<class ContainerAllocator>
struct DataType< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
{
  static const char* value()
  {
    return "nmea_msgs/GpgsvSatellite";
  }

  static const char* value(const ::nmea_msgs::GpgsvSatellite_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# Satellite data structure used in GPGSV messages\n\
\n\
# PRN number of the satellite\n\
# GPS = 1..32\n\
# SBAS = 33..64\n\
# GLO = 65..96\n\
uint8 prn\n\
\n\
# Elevation, degrees. Maximum 90\n\
uint8 elevation\n\
\n\
# Azimuth, True North degrees. [0, 359]\n\
uint16 azimuth\n\
\n\
# Signal to noise ratio, 0-99 dB. -1 when null in NMEA sentence (not tracking)\n\
int8 snr\n\
";
  }

  static const char* value(const ::nmea_msgs::GpgsvSatellite_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.prn);
      stream.next(m.elevation);
      stream.next(m.azimuth);
      stream.next(m.snr);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GpgsvSatellite_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::nmea_msgs::GpgsvSatellite_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::nmea_msgs::GpgsvSatellite_<ContainerAllocator>& v)
  {
    s << indent << "prn: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.prn);
    s << indent << "elevation: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.elevation);
    s << indent << "azimuth: ";
    Printer<uint16_t>::stream(s, indent + "  ", v.azimuth);
    s << indent << "snr: ";
    Printer<int8_t>::stream(s, indent + "  ", v.snr);
  }
};

} // namespace message_operations
} // namespace ros

#endif // NMEA_MSGS_MESSAGE_GPGSVSATELLITE_H
