// Generated by gencpp from file automotive_navigation_msgs/GetImageForMapTile.msg
// DO NOT EDIT!


#ifndef AUTOMOTIVE_NAVIGATION_MSGS_MESSAGE_GETIMAGEFORMAPTILE_H
#define AUTOMOTIVE_NAVIGATION_MSGS_MESSAGE_GETIMAGEFORMAPTILE_H

#include <ros/service_traits.h>


#include <automotive_navigation_msgs/GetImageForMapTileRequest.h>
#include <automotive_navigation_msgs/GetImageForMapTileResponse.h>


namespace automotive_navigation_msgs
{

struct GetImageForMapTile
{

typedef GetImageForMapTileRequest Request;
typedef GetImageForMapTileResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct GetImageForMapTile
} // namespace automotive_navigation_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::automotive_navigation_msgs::GetImageForMapTile > {
  static const char* value()
  {
    return "a619e5a4e3af6e680da86e0d146acebe";
  }

  static const char* value(const ::automotive_navigation_msgs::GetImageForMapTile&) { return value(); }
};

template<>
struct DataType< ::automotive_navigation_msgs::GetImageForMapTile > {
  static const char* value()
  {
    return "automotive_navigation_msgs/GetImageForMapTile";
  }

  static const char* value(const ::automotive_navigation_msgs::GetImageForMapTile&) { return value(); }
};


// service_traits::MD5Sum< ::automotive_navigation_msgs::GetImageForMapTileRequest> should match 
// service_traits::MD5Sum< ::automotive_navigation_msgs::GetImageForMapTile > 
template<>
struct MD5Sum< ::automotive_navigation_msgs::GetImageForMapTileRequest>
{
  static const char* value()
  {
    return MD5Sum< ::automotive_navigation_msgs::GetImageForMapTile >::value();
  }
  static const char* value(const ::automotive_navigation_msgs::GetImageForMapTileRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::automotive_navigation_msgs::GetImageForMapTileRequest> should match 
// service_traits::DataType< ::automotive_navigation_msgs::GetImageForMapTile > 
template<>
struct DataType< ::automotive_navigation_msgs::GetImageForMapTileRequest>
{
  static const char* value()
  {
    return DataType< ::automotive_navigation_msgs::GetImageForMapTile >::value();
  }
  static const char* value(const ::automotive_navigation_msgs::GetImageForMapTileRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::automotive_navigation_msgs::GetImageForMapTileResponse> should match 
// service_traits::MD5Sum< ::automotive_navigation_msgs::GetImageForMapTile > 
template<>
struct MD5Sum< ::automotive_navigation_msgs::GetImageForMapTileResponse>
{
  static const char* value()
  {
    return MD5Sum< ::automotive_navigation_msgs::GetImageForMapTile >::value();
  }
  static const char* value(const ::automotive_navigation_msgs::GetImageForMapTileResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::automotive_navigation_msgs::GetImageForMapTileResponse> should match 
// service_traits::DataType< ::automotive_navigation_msgs::GetImageForMapTile > 
template<>
struct DataType< ::automotive_navigation_msgs::GetImageForMapTileResponse>
{
  static const char* value()
  {
    return DataType< ::automotive_navigation_msgs::GetImageForMapTile >::value();
  }
  static const char* value(const ::automotive_navigation_msgs::GetImageForMapTileResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // AUTOMOTIVE_NAVIGATION_MSGS_MESSAGE_GETIMAGEFORMAPTILE_H