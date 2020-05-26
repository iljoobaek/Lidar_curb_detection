// Generated by gencpp from file jsk_gui_msgs/Query.msg
// DO NOT EDIT!


#ifndef JSK_GUI_MSGS_MESSAGE_QUERY_H
#define JSK_GUI_MSGS_MESSAGE_QUERY_H

#include <ros/service_traits.h>


#include <jsk_gui_msgs/QueryRequest.h>
#include <jsk_gui_msgs/QueryResponse.h>


namespace jsk_gui_msgs
{

struct Query
{

typedef QueryRequest Request;
typedef QueryResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct Query
} // namespace jsk_gui_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::jsk_gui_msgs::Query > {
  static const char* value()
  {
    return "9c540946b387ce7acce975ce4dfac8ad";
  }

  static const char* value(const ::jsk_gui_msgs::Query&) { return value(); }
};

template<>
struct DataType< ::jsk_gui_msgs::Query > {
  static const char* value()
  {
    return "jsk_gui_msgs/Query";
  }

  static const char* value(const ::jsk_gui_msgs::Query&) { return value(); }
};


// service_traits::MD5Sum< ::jsk_gui_msgs::QueryRequest> should match 
// service_traits::MD5Sum< ::jsk_gui_msgs::Query > 
template<>
struct MD5Sum< ::jsk_gui_msgs::QueryRequest>
{
  static const char* value()
  {
    return MD5Sum< ::jsk_gui_msgs::Query >::value();
  }
  static const char* value(const ::jsk_gui_msgs::QueryRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::jsk_gui_msgs::QueryRequest> should match 
// service_traits::DataType< ::jsk_gui_msgs::Query > 
template<>
struct DataType< ::jsk_gui_msgs::QueryRequest>
{
  static const char* value()
  {
    return DataType< ::jsk_gui_msgs::Query >::value();
  }
  static const char* value(const ::jsk_gui_msgs::QueryRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::jsk_gui_msgs::QueryResponse> should match 
// service_traits::MD5Sum< ::jsk_gui_msgs::Query > 
template<>
struct MD5Sum< ::jsk_gui_msgs::QueryResponse>
{
  static const char* value()
  {
    return MD5Sum< ::jsk_gui_msgs::Query >::value();
  }
  static const char* value(const ::jsk_gui_msgs::QueryResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::jsk_gui_msgs::QueryResponse> should match 
// service_traits::DataType< ::jsk_gui_msgs::Query > 
template<>
struct DataType< ::jsk_gui_msgs::QueryResponse>
{
  static const char* value()
  {
    return DataType< ::jsk_gui_msgs::Query >::value();
  }
  static const char* value(const ::jsk_gui_msgs::QueryResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // JSK_GUI_MSGS_MESSAGE_QUERY_H