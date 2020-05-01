// Generated by gencpp from file spatio_temporal_voxel_layer/SaveGrid.msg
// DO NOT EDIT!


#ifndef SPATIO_TEMPORAL_VOXEL_LAYER_MESSAGE_SAVEGRID_H
#define SPATIO_TEMPORAL_VOXEL_LAYER_MESSAGE_SAVEGRID_H

#include <ros/service_traits.h>


#include <spatio_temporal_voxel_layer/SaveGridRequest.h>
#include <spatio_temporal_voxel_layer/SaveGridResponse.h>


namespace spatio_temporal_voxel_layer
{

struct SaveGrid
{

typedef SaveGridRequest Request;
typedef SaveGridResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct SaveGrid
} // namespace spatio_temporal_voxel_layer


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::spatio_temporal_voxel_layer::SaveGrid > {
  static const char* value()
  {
    return "793bb7917a99799f9a78324a148a1c17";
  }

  static const char* value(const ::spatio_temporal_voxel_layer::SaveGrid&) { return value(); }
};

template<>
struct DataType< ::spatio_temporal_voxel_layer::SaveGrid > {
  static const char* value()
  {
    return "spatio_temporal_voxel_layer/SaveGrid";
  }

  static const char* value(const ::spatio_temporal_voxel_layer::SaveGrid&) { return value(); }
};


// service_traits::MD5Sum< ::spatio_temporal_voxel_layer::SaveGridRequest> should match 
// service_traits::MD5Sum< ::spatio_temporal_voxel_layer::SaveGrid > 
template<>
struct MD5Sum< ::spatio_temporal_voxel_layer::SaveGridRequest>
{
  static const char* value()
  {
    return MD5Sum< ::spatio_temporal_voxel_layer::SaveGrid >::value();
  }
  static const char* value(const ::spatio_temporal_voxel_layer::SaveGridRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::spatio_temporal_voxel_layer::SaveGridRequest> should match 
// service_traits::DataType< ::spatio_temporal_voxel_layer::SaveGrid > 
template<>
struct DataType< ::spatio_temporal_voxel_layer::SaveGridRequest>
{
  static const char* value()
  {
    return DataType< ::spatio_temporal_voxel_layer::SaveGrid >::value();
  }
  static const char* value(const ::spatio_temporal_voxel_layer::SaveGridRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::spatio_temporal_voxel_layer::SaveGridResponse> should match 
// service_traits::MD5Sum< ::spatio_temporal_voxel_layer::SaveGrid > 
template<>
struct MD5Sum< ::spatio_temporal_voxel_layer::SaveGridResponse>
{
  static const char* value()
  {
    return MD5Sum< ::spatio_temporal_voxel_layer::SaveGrid >::value();
  }
  static const char* value(const ::spatio_temporal_voxel_layer::SaveGridResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::spatio_temporal_voxel_layer::SaveGridResponse> should match 
// service_traits::DataType< ::spatio_temporal_voxel_layer::SaveGrid > 
template<>
struct DataType< ::spatio_temporal_voxel_layer::SaveGridResponse>
{
  static const char* value()
  {
    return DataType< ::spatio_temporal_voxel_layer::SaveGrid >::value();
  }
  static const char* value(const ::spatio_temporal_voxel_layer::SaveGridResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // SPATIO_TEMPORAL_VOXEL_LAYER_MESSAGE_SAVEGRID_H
