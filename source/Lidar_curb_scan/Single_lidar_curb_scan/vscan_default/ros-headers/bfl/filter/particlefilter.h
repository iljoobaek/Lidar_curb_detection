// $Id: particlefilter.h 29830 2009-01-14 15:10:41Z kgadeyne $

 /***************************************************************************
 *   This library is free software; you can redistribute it and/or         *
 *   modify it under the terms of the GNU General Public                   *
 *   License as published by the Free Software Foundation;                 *
 *   version 2 of the License.                                             *
 *                                                                         *
 *   As a special exception, you may use this file as part of a free       *
 *   software library without restriction.  Specifically, if other files   *
 *   instantiate templates or use macros or inline functions from this     *
 *   file, or you compile this file and link it with other files to        *
 *   produce an executable, this file does not by itself cause the         *
 *   resulting executable to be covered by the GNU General Public          *
 *   License.  This exception does not however invalidate any other        *
 *   reasons why the executable file might be covered by the GNU General   *
 *   Public License.                                                       *
 *                                                                         *
 *   This library is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU     *
 *   Lesser General Public License for more details.                       *
 *                                                                         *
 *   You should have received a copy of the GNU General Public             *
 *   License along with this library; if not, write to the Free Software   *
 *   Foundation, Inc., 59 Temple Place,                                    *
 *   Suite 330, Boston, MA  02111-1307  USA                                *
 *                                                                         *
 ***************************************************************************/

#ifndef __PARTICLE_FILTER__
#define __PARTICLE_FILTER__

#include "filter.h"
#include "../pdf/conditionalpdf.h"
#include "../pdf/mcpdf.h"

// RS stands for Resample Scheme
// TODO: Work this better out
#define DEFAULT_RS MULTINOMIAL_RS // Default scheme = MULTINOMIAL
#define MULTINOMIAL_RS 0
/* Carpenter, Clifford and Fearnhead:
   Efficient implementation of particle
   filters for non-linear systems.

   See eg.

   @Article{        gordon93,
   author = 	 {Gordon, Neil and Salmond, D. J. and Smith, A. F. M.},
   title = 	 {Novel approach to nonlinear/non-Gaussian state estimation},
   journal = 	 {IEE Proceedings-F},
   year = 	 {1993},
   volume =	 {140},
   number =	 {2},
   pages =	 {107--113},
   annote =	 {Multinomial Sampling}}
*/

#define SYSTEMATIC_RS 1
// A lot of possible systematic approaches to resampling are described
// in literature.  The goal is to reduce the Monte Carlo Variance of
// Particle filters.  One example is
/*
@Article{        carpenter99-improved,
  author = 	 {Carpenter, J. and Clifford, P. and Fearnhead, P.},
  title = 	 {An {I}mproved {P}article {F}ilter for {N}on-linear {P}roblems},
  journal = 	 {Radar, Sonar and Navigation, IEE Proceedings -},
  year = 	 {1999},
  volume =	 {146},
  number =	 {1},
  pages =	 {2--7},
  month =	 {February},
  annote =	 {Describes systematic resampling, variance reduction}
}
*/

// KG TODO Check if systematic resampling = deterministic resampling
// as described by kitagawa

#define STRATIFIED_RS 2
// Generate N samples not uniformly on [0,1] but generate 1 sample
// uniformly in for each "stratum" u_j ~ U(j-1/N,j/N)
// Variance reduction!
/*
  @Article{         kitagawa96,
  author = 	 {Kitagawa, G.},
  title = 	 {{M}onte {C}arlo filter and smoother for non-{G}aussian nonlinear state space models },
  journal = 	 {Journal of Computational and Graphical Statistics},
  year = 	 {1996},
  volume =	 {5},
  number =	 {1},
  pages =	 {1--25},
  annote =	 {describes deterministic and stratified resampling}
  }
*/

#define RESIDUAL_RS 3
// sample "deterministically" the integer part of N \times w_j , the
// perform multinomial sampling for the resulting N - \sum [N \times
// w_j] where [] denotes the integer part.
/*
  See eg. p.19
  @Article{         liuchen98,
  author = 	 {Liu, J. S. and Chen, R.},
  title = 	 {Sequential {M}onte {C}arlo methods for dynamic systems},
  journal = 	 {Journal of the American Statistical Association},
  year = 	 {1998},
  volume =	 {93},
  pages =	 {1032--1044}}
*/

#define MINIMUM_VARIANCE_RS 4
/*
  See eg.
  @TechReport{     carpenter99,
  author = 	 {Carpenter, J. and Clifford, P. and Fearnhead, P.},
  title = 	 {Building robust simulation-based filters for evolving data sets},
  institution =  {Department of Statistics, University of Oxford},
  year = 	 {99},
  note =	 {\url{http://www.stats.ox.ac.uk/pub/clifford/Particle_Filters/}}}
*/

namespace BFL
{

  /// Virtual Class representing all particle filters
  /** This is a virtual class representing the family of all particle
      filters.  Particle filters are filters in which the Posterior
      density is represented by a set of particles (aka (weighted)
      samples).  In other words, the posterior density is a Monte Carlo
      Pdf (class MCPdf)

      However, the updating of the Posterior density can still be done
      in several ways, that's why the System and Measurement update
      members are still pure virtual functions.

      This class is the base class for all sorts of particle filters.

      @todo:  Actually all particle filters represented by this class
      are of the "Sequential importance sampling methods" type.  Typical
      of those methods is the so called Proposal density.  In theory it
      would be possible to create Filters using a recursive version of
      other Monte Carlo methods (eg. MCMC methods), although I am not
      aware of any of these (due to the increased complexity).

      @see MCPdf
      @see Sample
      @see WeightedSample

      @bug Resampling is not implemented generically enough yet.
      There's only the possibility to choose between static period
      resampling and dynamic resampling as proposed by Jun Liu.
      The correct way of implementing this would be to create a
      virtual function that has to be implemented by the user, but
      this creates more hassle for the user (a different particle
      filter for each scheme).

  */
  template <typename StateVar, typename MeasVar> class ParticleFilter
    : public Filter<StateVar,MeasVar>
    {
    protected:
      virtual bool UpdateInternal(SystemModel<StateVar>* const sysmodel,
				  const StateVar& u,
				  MeasurementModel<MeasVar,StateVar>* const measmodel,
				  const MeasVar& z,
				  const StateVar& s);

      /// Pointer to the Proposal Density
      /** Every particle filter (or more correct: every Sequential
	  Importance Sampling method) uses a proposal density to do the
	  forward sampling step
      */
      ConditionalPdf<StateVar,StateVar> * _proposal;

      /// While updating use sample<StateVar>
      WeightedSample<StateVar>  _sample;
      /// While updating store list of old samples
      vector<WeightedSample<StateVar> > _old_samples;
      /// While updating store list of new samples
      vector<WeightedSample<StateVar> > _new_samples;
      /// While resampling
      vector<Sample<StateVar> > _new_samples_unweighted;
      /// Iterator for old list of samples
      typename vector<WeightedSample<StateVar> >::iterator _os_it;
      /// Iterator for new list of samples
      typename vector<WeightedSample<StateVar> >::iterator _ns_it;

      /// Number of timestep between resampling from the Posterior Pdf.
      /** By choosing this period, one can avoid numerical instability (aka
	  Degeneration of the particle filter
      */
      int _resamplePeriod;

      /// Threshold used when dynamic resampling
      double _resampleThreshold;

      /// Which resample algorithm (see top of particle.h for #defines)
      int _resampleScheme;

      /// Dynamic resampling or fixed period resampling?
      bool _dynamicResampling;

      /// Proposal depends on last measurement?
      bool _proposal_depends_on_meas;

      /// created own post
      bool _created_post;

      /// Proposal step
      /** Implementation of proposal step
	  @param sysmodel pointer to the used system model
	  @param u input param for proposal density
	  @param measmodel pointer to the used measurementmodel
	  @param z measurement param for proposal density
	  @param s sensor param for proposal density
	  @bug Make sampling method variable.  See implementation.
      */
      virtual bool ProposalStepInternal(SystemModel<StateVar> * const sysmodel,
	                                const StateVar & u,
	                                MeasurementModel<MeasVar,StateVar> * const measmodel,
					const MeasVar & z,
					const StateVar & s);

      /// Update Weights
      /** @param sysmodel pointer to the used system model
	  @param u input param for proposal density
	  @param measmodel pointer to the used measurementmodel
	  @param z measurement param for proposal density
	  @param s sensor param for proposal density
      */
      virtual bool UpdateWeightsInternal(SystemModel<StateVar> * const sysmodel,
					 const StateVar & u,
					 MeasurementModel<MeasVar,StateVar> * const measmodel,
					 const MeasVar & z,
					 const StateVar & s);

      /// Resample if necessary
      /** @bug let the user implement her/his own resamplescheme
       */
      virtual bool DynamicResampleStep();

      /// Resample if wanted
      /** @bug let the user implement her/his own resamplescheme
       */
      virtual bool StaticResampleStep();

      /// Actual Resampling happens here;
      virtual bool Resample();

    public:
      /// Constructor
      /** @pre you created the necessary models and the prior
	  @param prior pointer to the Monte Carlo Pdf prior density
	  @param proposal pointer to the proposal density to use
	  @param resampleperiod fixed resampling period (if desired)
	  @param resamplethreshold threshold used when dynamic resampling
	  @param resamplescheme resampling scheme, see header file for
	  different defines and their meaning
	  @bug prior should be of type pdf and not mcpdf.  See also
	  notes with implementation
	  @bug let the user implement her/his own resamplescheme
      */
      ParticleFilter(MCPdf<StateVar> * prior,
		     ConditionalPdf<StateVar,StateVar> * proposal,
		     int resampleperiod = 0,
		     double resamplethreshold = 0,
		     int resamplescheme = DEFAULT_RS);


      /// Constructor
      /** @pre you created the necessary models and the prior
	  @param prior pointer to the Monte Carlo Pdf prior density
	  @param post pointer to the Monte Carlo Pdf post density
	  @param proposal pointer to the proposal density to use
	  @param resampleperiod fixed resampling period (if desired)
	  @param resamplethreshold threshold used when dynamic resampling
	  @param resamplescheme resampling scheme, see header file for
	  different defines and their meaning
	  @bug prior should be of type pdf and not mcpdf.  See also
	  notes with implementation
	  @bug let the user implement her/his own resamplescheme
      */
      ParticleFilter(MCPdf<StateVar> * prior,
		     MCPdf<StateVar> * post,
		     ConditionalPdf<StateVar,StateVar> * proposal,
		     int resampleperiod = 0,
		     double resamplethreshold = 0,
		     int resamplescheme = DEFAULT_RS);

      /// Destructor
      virtual ~ParticleFilter();
      /// Copy Constructor
      /** @bug implementation probably contains a bug
       */
      ParticleFilter(const ParticleFilter<StateVar,MeasVar> & filt);

      /// Set the proposal density
      /** @param cpdf the new proposal density.  The order of the
	  conditional arguments is fixed and should be: x (state), u
	  (input), z (measurement), s (sensor param).  Off course all
	  of them are optional
       */
      virtual void ProposalSet(ConditionalPdf<StateVar,StateVar>* const cpdf);

      /// Get a pointer to the proposal density
      /** @return a pointer to the proposal density
       */
      ConditionalPdf<StateVar,StateVar> * ProposalGet();

      // implement virtual function
      virtual MCPdf<StateVar> * PostGet();
    };

#include "particlefilter.cpp"

} // End namespace BFL

#endif // __PARTICLE_FILTER__
