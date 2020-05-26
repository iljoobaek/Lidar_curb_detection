// Copyright (C) 2003 Klaas Gadeyne <first dot last at gmail dot com>
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
// $Id: bootstrapfilter.cpp 29495 2008-08-13 12:57:49Z tdelaet $

#include "bootstrapfilter.h"
#include "../sample/weightedsample.h"

#define StateVar SVar
#define MeasVar MVar

template <typename SVar, typename MVar>
BootstrapFilter<SVar,MVar>::BootstrapFilter(MCPdf<SVar> * prior,
					    int resampleperiod,
					    double resamplethreshold,
					    int resamplescheme)
  : ParticleFilter<SVar,MVar>(prior,NULL,resampleperiod,
				     resamplethreshold,
				     resamplescheme)
{
  // for a bootstrapfilter, the proposal does not depend on the
  // measurement
  this->_proposal_depends_on_meas = false;
}


template <typename SVar, typename MVar>
BootstrapFilter<SVar,MVar>::BootstrapFilter(MCPdf<SVar> * prior,
					    MCPdf<SVar> * post,
					    int resampleperiod,
					    double resamplethreshold,
					    int resamplescheme)
  : ParticleFilter<SVar,MVar>(prior,post,NULL,resampleperiod,
				     resamplethreshold,
				     resamplescheme)
{
  // for a bootstrapfilter, the proposal does not depend on the
  // measurement
  this->_proposal_depends_on_meas = false;
}




template <typename SVar, typename MVar>
BootstrapFilter<SVar,MVar>::~BootstrapFilter(){}

template <typename SVar, typename MVar> bool
BootstrapFilter<SVar,MVar>::UpdateInternal(SystemModel<SVar>* const sysmodel,
					   const SVar& u,
					   MeasurementModel<MVar,SVar>* const measmodel,
					   const MVar& z,
					   const SVar& s)
{
  bool result = true;

  if (sysmodel != NULL){
    this->ProposalSet(sysmodel->SystemPdfGet());
    result = this->ParticleFilter<SVar,MVar>::UpdateInternal(sysmodel,u,NULL,z,s) && result;
  }
  if (measmodel != NULL)
    result = this->ParticleFilter<SVar,MVar>::UpdateInternal(NULL,u,measmodel,z,s) && result;

  return result;
}

