// $Id: discreteconditionalpdf.h 29890 2009-02-02 10:22:01Z tdelaet $
// Copyright (C) 2002 Klaas Gadeyne <first dot last at gmail dot com>
//               2008 Tinne De Laet <first dot last at mech dot kuleuven dot be>
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

#ifndef __DISCRETE_CONDITIONAL_PDF__
#define __DISCRETE_CONDITIONAL_PDF__

#include "conditionalpdf.h"

namespace BFL
{

  /// Abstract Class representing all _FULLY_ Discrete Conditional PDF's
  /** This class inherits only from ConditionalPdf (not from
      DiscretePdf, avoiding a circular  class structure


      <pre>
            ------
            |    |
            ------
           /      \
      -----        -----
      |   |        |   |
      -----        -----
           \      /
            ------
	    |    |
	    ------
      </pre>

      @todo Check if this is the best way to implement this.
      @note that the name of this class could be better chosen.
      Something like Discrete-DiscreteConditionalPdf would maybe be more
      clear (???), but quite long...
      @see ConditionalPdf
  */
  class DiscreteConditionalPdf : public ConditionalPdf<int, int>
    {
    protected:
      /// number of discrete states
      unsigned int _num_states;

      /// Pointer to the probability values
      /** For now we implement this using a simple row of doubles, this should
	  probably become a tensor in the future
      */
      double * _probability_p;
      /// "Possible discrete states" of all the conditional arguments
      int * _cond_arg_dims_p;

      /// Total dimension of the likelihoodtable
      int _total_dimension;

      /// Get the correct index in the row of doubles (double * probability)
      int IndexGet(const int& input, const std::vector<int>& condargs) const;

      // variables to prevent memory allocation on the heap during call of
      //SampleFrom
      mutable std::vector<double> _probs;
      mutable std::vector<double> _valuelist;

    public:
      /// Constructor
      /** @pre The number of elements of cond_arg_dimensions should be
	  equal to num_conditional_arguments, otherwise -> Segfaults
	  @param num_states int representing the number of possible states
	  @param num_conditional_arguments the number of arguments behind
	  the |
	  @param cond_arg_dimensions[] possible number of states of the
	  different conditional arguments
	  @see ConditionalPdf
	  @todo Get cleaner api and implementation
      */
      DiscreteConditionalPdf(int num_states=1,
			     int num_conditional_arguments=1,
			     int cond_arg_dimensions[] = NULL);
      /// Copy constructor
      DiscreteConditionalPdf(const DiscreteConditionalPdf & pdf);
      /// Destructor
      virtual ~DiscreteConditionalPdf();

      ///Clone function
      virtual DiscreteConditionalPdf* Clone() const;

      /// Get the number of discrete states
      unsigned int NumStatesGet()const;

      // Redefine all pure virtuals!
      Probability ProbabilityGet(const int& input) const;
      virtual bool SampleFrom (Sample<int>& one_sample, int method, void * args) const;
      virtual bool SampleFrom (vector<Sample<int> >& list_samples, int num_samples, int method, void * args) const;

      /// Set the probability (Typical for discrete Pdf's)
      void ProbabilitySet(const double& prob, const int& input, const std::vector<int>& condargs) const;


    };

} // End namespace BFL

#endif // __DISCRETE_CONDITIONAL_PDF__
