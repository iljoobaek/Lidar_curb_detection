// $Id: iteratedextendedkalmanfilter.h 30022 2009-03-06 15:10:30Z kgadeyne $
// Copyright (C) 2002 Klaas Gadeyne <first dot last at gmail dot com>
//                    Wim Meeussen  <wim dot meeussen at mech dot kuleuven dot ac dot be>
//                    Tinne De Laet  <tinne dot delaet at mech dot kuleuven dot be>
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//

#ifndef __ITERATED_EXTENDED_KALMAN_FILTER__
#define __ITERATED_EXTENDED_KALMAN_FILTER__

#include "kalmanfilter.h"
#include "../pdf/conditionalpdf.h"
#include "../pdf/gaussian.h"
#include "innovationCheck.h"
# include <map>

namespace BFL
{
  /** This is a class implementing the Kalman Filter (KF) class for
      Iterated Extended Kalman Filters.

      The System- and MeasurementUpdate equasions are not linear, and
      will be approximated by local linearisations.

      @see KalmanFilter
  */
  class IteratedExtendedKalmanFilter : public KalmanFilter
    {
    protected:
      virtual void SysUpdate(SystemModel<MatrixWrapper::ColumnVector>* const sysmodel,
			     const MatrixWrapper::ColumnVector& u);
      virtual void MeasUpdate(MeasurementModel<MatrixWrapper::ColumnVector,MatrixWrapper::ColumnVector>* const measmodel,
			      const MatrixWrapper::ColumnVector& z,
			      const MatrixWrapper::ColumnVector& s);

    public:
      /** Constructor
	  @pre you created the prior
	  @param prior pointer to the Monte Carlo Pdf prior density
	  @param nr_it the number of iterations in one update
	  @param innov pointer to InnovationCheck (to end the iterations of the innovation is too small)
      */
      IteratedExtendedKalmanFilter(Gaussian* prior, unsigned int nr_it=1, InnovationCheck* innov = NULL);

      /// Destructor
      virtual ~IteratedExtendedKalmanFilter();

      /// Function to allocate memory needed during the measurement update,
      //  For realtime use, this function should be called before calling measUpdate
      /*  @param vector containing the dimension of the measurement models which are
          going to be used
      */
      void AllocateMeasModelIExt( const vector<unsigned int>& meas_dimensions);

      /// Function to allocate memory needed during the measurement update
      //  For realtime use, this function should be called before calling measUpdate
      /*  @param dimension of the measurement models which is
          going to be used
      */
      void AllocateMeasModelIExt( const unsigned int& meas_dimensions);

    private:
      /// number of iterations for iterated extended kalman filter
      unsigned int _nr_iterations;
      /// pointer to InnovationCheck (to end the iterations if the innovation is too small)
      InnovationCheck* _innovationChecker;

      struct MeasUpdateVariablesIExt
      {
        SymmetricMatrix _R_i;
        Matrix          _K_i;
        Matrix          _H_i;
        ColumnVector    _Z_i;
        MeasUpdateVariablesIExt() {};
        MeasUpdateVariablesIExt(unsigned int meas_dimension, unsigned int state_dimension):
          _R_i(meas_dimension)
        , _K_i(state_dimension,meas_dimension)
        , _H_i(meas_dimension,state_dimension)
        , _Z_i(meas_dimension)
         {};
      }; //struct

      // variables to avoid allocation on the heap
      ColumnVector _x;
      ColumnVector _x_i;
      ColumnVector _x_i_prev;
      ColumnVector _J;
      ColumnVector _innovation;
      Matrix    _F;
      SymmetricMatrix _Q;
      SymmetricMatrix _P_Matrix;
      Matrix          _S_i;
      std::map<unsigned int, MeasUpdateVariablesIExt> _mapMeasUpdateVariablesIExt;
      std::map<unsigned int, MeasUpdateVariablesIExt>::iterator _mapMeasUpdateVariablesIExt_it;
    };  // class

} // End namespace BFL

#endif // __ITERATED_EXTENDED_KALMAN_FILTER__

