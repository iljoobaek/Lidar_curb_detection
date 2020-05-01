// Generated by gencpp from file dynamic_tf_publisher/DeleteTF.msg
// DO NOT EDIT!


#ifndef DYNAMIC_TF_PUBLISHER_MESSAGE_DELETETF_H
#define DYNAMIC_TF_PUBLISHER_MESSAGE_DELETETF_H

#include <ros/service_traits.h>


#include <dynamic_tf_publisher/DeleteTFRequest.h>
#include <dynamic_tf_publisher/DeleteTFResponse.h>


namespace dynamic_tf_publisher
{

struct DeleteTF
{

typedef DeleteTFRequest Request;
typedef DeleteTFResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct DeleteTF
} // namespace dynamic_tf_publisher


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::dynamic_tf_publisher::DeleteTF > {
  static const char* value()
  {
    return "d7be0bb39af8fb9129d5a76e6b63a290";
  }

  static const char* value(const ::dynamic_tf_publisher::DeleteTF&) { return value(); }
};

template<>
struct DataType< ::dynamic_tf_publisher::DeleteTF > {
  static const char* value()
  {
    return "dynamic_tf_publisher/DeleteTF";
  }

  static const char* value(const ::dynamic_tf_publisher::DeleteTF&) { return value(); }
};


// service_traits::MD5Sum< ::dynamic_tf_publisher::DeleteTFRequest> should match 
// service_traits::MD5Sum< ::dynamic_tf_publisher::DeleteTF > 
template<>
struct MD5Sum< ::dynamic_tf_publisher::DeleteTFRequest>
{
  static const char* value()
  {
    return MD5Sum< ::dynamic_tf_publisher::DeleteTF >::value();
  }
  static const char* value(const ::dynamic_tf_publisher::DeleteTFRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::dynamic_tf_publisher::DeleteTFRequest> should match 
// service_traits::DataType< ::dynamic_tf_publisher::DeleteTF > 
template<>
struct DataType< ::dynamic_tf_publisher::DeleteTFRequest>
{
  static const char* value()
  {
    return DataType< ::dynamic_tf_publisher::DeleteTF >::value();
  }
  static const char* value(const ::dynamic_tf_publisher::DeleteTFRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::dynamic_tf_publisher::DeleteTFResponse> should match 
// service_traits::MD5Sum< ::dynamic_tf_publisher::DeleteTF > 
template<>
struct MD5Sum< ::dynamic_tf_publisher::DeleteTFResponse>
{
  static const char* value()
  {
    return MD5Sum< ::dynamic_tf_publisher::DeleteTF >::value();
  }
  static const char* value(const ::dynamic_tf_publisher::DeleteTFResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::dynamic_tf_publisher::DeleteTFResponse> should match 
// service_traits::DataType< ::dynamic_tf_publisher::DeleteTF > 
template<>
struct DataType< ::dynamic_tf_publisher::DeleteTFResponse>
{
  static const char* value()
  {
    return DataType< ::dynamic_tf_publisher::DeleteTF >::value();
  }
  static const char* value(const ::dynamic_tf_publisher::DeleteTFResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // DYNAMIC_TF_PUBLISHER_MESSAGE_DELETETF_H
