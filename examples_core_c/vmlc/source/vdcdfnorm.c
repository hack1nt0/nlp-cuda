/*******************************************************************************
* Copyright 2001-2015 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/*
!  Content:
!    vdCdfNorm  Example Program Text
!******************************************************************************/

#include <stdio.h>
#include "mkl_vml.h"

#include "_rms.h"

int main()
{
  double dA[10],dB[10];
  double dBha0[10],dBha1[10],dBha2[10];
  double           dBla1[10],dBla2[10];
  double           dBep1[10],dBep2[10];
  float CurRMS,MaxRMS=0.0;

  MKL_INT i=0,vec_len=10;

  dA[0]=-8.0000;
  dA[1]=-6.5555;
  dA[2]=-5.1111;
  dA[3]=-3.6666;
  dA[4]=-2.2222;
  dA[5]=-0.7777;
  dA[6]=0.6666;
  dA[7]=2.1111;
  dA[8]=3.5555;
  dA[9]=5.0000;
  dB[0]=6.22096057427178e-16;
  dB[1]=2.77278746107643e-11;
  dB[2]=1.60144189579716e-07;
  dB[3]=1.22898411922205e-04;
  dB[4]=1.31348962351314e-02;
  dB[5]=2.18372946100606e-01;
  dB[6]=7.47486165446128e-01;
  dB[7]=9.82618141277281e-01;
  dB[8]=9.99811369527550e-01;
  dB[9]=9.99999713348428e-01;

  vdCdfNorm(vec_len,dA,dBha0);

  vmdCdfNorm(vec_len,dA,dBep1,VML_EP);

  vmlSetMode(VML_EP);
  vdCdfNorm(vec_len,dA,dBep2);

  vmdCdfNorm(vec_len,dA,dBla1,VML_LA);

  vmlSetMode(VML_LA);
  vdCdfNorm(vec_len,dA,dBla2);

  vmdCdfNorm(vec_len,dA,dBha1,VML_HA);

  vmlSetMode(VML_HA);
  vdCdfNorm(vec_len,dA,dBha2);

  for(i=0;i<10;i++) {
    if(dBha0[i]!=dBha1[i] || dBha1[i]!=dBha2[i]) {
      printf("Error! Difference between vdCdfNorm and vmdCdfNorm in VML_HA mode detected.\n");
      return 1;
    }

    if(dBla1[i]!=dBla2[i]) {
      printf("Error! Difference between vdCdfNorm and vmdCdfNorm in VML_LA mode detected.\n");
      return 1;
    }

    if(dBep1[i]!=dBep2[i]) {
      printf("Error! Difference between vdCdfNorm and vmdCdfNorm in VML_EP mode detected.\n");
      return 1;
    }
  }

  printf("vdCdfNorm test/example program\n\n");
  printf("           Argument                     vdCdfNorm\n");
  printf("===============================================================================\n");
  for(i=0;i<10;i++) {
    printf("% 25.14f % 25.14e\n",dA[i],dBha0[i]);
    CurRMS=drelerr(dB[i],dBha0[i]);
    if(CurRMS>MaxRMS) MaxRMS=CurRMS;
  }
  printf("\n");
  if(MaxRMS>=1e-14) {
    printf("Error! Relative accuracy is %.16f\n",MaxRMS);
    return 1;
  }
  else {
    printf("Relative accuracy is %.16f\n",MaxRMS);
  }

  return 0;
}