// Generated by gencpp from file jsk_recognition_msgs/SetTemplate.msg
// DO NOT EDIT!


#ifndef JSK_RECOGNITION_MSGS_MESSAGE_SETTEMPLATE_H
#define JSK_RECOGNITION_MSGS_MESSAGE_SETTEMPLATE_H

#include <ros/service_traits.h>


#include <jsk_recognition_msgs/SetTemplateRequest.h>
#include <jsk_recognition_msgs/SetTemplateResponse.h>


namespace jsk_recognition_msgs
{

struct SetTemplate
{

typedef SetTemplateRequest Request;
typedef SetTemplateResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SetTemplate
} // namespace jsk_recognition_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::jsk_recognition_msgs::SetTemplate > {
  static const char* value()
  {
    return "116fa80f27cbdfcd76d0b57a30ef79ec";
  }

  static const char* value(const ::jsk_recognition_msgs::SetTemplate&) { return value(); }
};

template<>
struct DataType< ::jsk_recognition_msgs::SetTemplate > {
  static const char* value()
  {
    return "jsk_recognition_msgs/SetTemplate";
  }

  static const char* value(const ::jsk_recognition_msgs::SetTemplate&) { return value(); }
};


// service_traits::MD5Sum< ::jsk_recognition_msgs::SetTemplateRequest> should match 
// service_traits::MD5Sum< ::jsk_recognition_msgs::SetTemplate > 
template<>
struct MD5Sum< ::jsk_recognition_msgs::SetTemplateRequest>
{
  static const char* value()
  {
    return MD5Sum< ::jsk_recognition_msgs::SetTemplate >::value();
  }
  static const char* value(const ::jsk_recognition_msgs::SetTemplateRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::jsk_recognition_msgs::SetTemplateRequest> should match 
// service_traits::DataType< ::jsk_recognition_msgs::SetTemplate > 
template<>
struct DataType< ::jsk_recognition_msgs::SetTemplateRequest>
{
  static const char* value()
  {
    return DataType< ::jsk_recognition_msgs::SetTemplate >::value();
  }
  static const char* value(const ::jsk_recognition_msgs::SetTemplateRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::jsk_recognition_msgs::SetTemplateResponse> should match 
// service_traits::MD5Sum< ::jsk_recognition_msgs::SetTemplate > 
template<>
struct MD5Sum< ::jsk_recognition_msgs::SetTemplateResponse>
{
  static const char* value()
  {
    return MD5Sum< ::jsk_recognition_msgs::SetTemplate >::value();
  }
  static const char* value(const ::jsk_recognition_msgs::SetTemplateResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::jsk_recognition_msgs::SetTemplateResponse> should match 
// service_traits::DataType< ::jsk_recognition_msgs::SetTemplate > 
template<>
struct DataType< ::jsk_recognition_msgs::SetTemplateResponse>
{
  static const char* value()
  {
    return DataType< ::jsk_recognition_msgs::SetTemplate >::value();
  }
  static const char* value(const ::jsk_recognition_msgs::SetTemplateResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // JSK_RECOGNITION_MSGS_MESSAGE_SETTEMPLATE_H