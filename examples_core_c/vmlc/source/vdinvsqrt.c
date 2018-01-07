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
!    vdInvSqrt  Example Program Text
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

  dA[0]=0.1000;
  dA[1]=1111.1999;
  dA[2]=2222.2999;
  dA[3]=3333.3999;
  dA[4]=4444.5000;
  dA[5]=5555.6000;
  dA[6]=6666.7000;
  dA[7]=7777.8000;
  dA[8]=8888.9000;
  dA[9]=10000.0000;
  dB[0]=3.1622776601683791e+000;
  dB[1]=2.9998801421833310e-002;
  dB[2]=2.1212832691552868e-002;
  dB[3]=1.7320335133005883e-002;
  dB[4]=1.4999906250878898e-002;
  dB[5]=1.3416354199689269e-002;
  dB[6]=1.2247418095408925e-002;
  dB[7]=1.1338917991834114e-002;
  dB[8]=1.0606595088678355e-002;
  dB[9]=1.0000000000000000e-002;

  vdInvSqrt(vec_len,dA,dBha0);

  vmdInvSqrt(vec_len,dA,dBep1,VML_EP);

  vmlSetMode(VML_EP);
  vdInvSqrt(vec_len,dA,dBep2);

  vmdInvSqrt(vec_len,dA,dBla1,VML_LA);

  vmlSetMode(VML_LA);
  vdInvSqrt(vec_len,dA,dBla2);

  vmdInvSqrt(vec_len,dA,dBha1,VML_HA);

  vmlSetMode(VML_HA);
  vdInvSqrt(vec_len,dA,dBha2);

  for(i=0;i<10;i++) {
    if(dBha0[i]!=dBha1[i] || dBha1[i]!=dBha2[i]) {
      printf("Error! Difference between vdInvSqrt and vmdInvSqrt in VML_HA mode detected.\n");
      return 1;
    }

    if(dBla1[i]!=dBla2[i]) {
      printf("Error! Difference between vdInvSqrt and vmdInvSqrt in VML_LA mode detected.\n");
      return 1;
    }

    if(dBep1[i]!=dBep2[i]) {
      printf("Error! Difference between vdInvSqrt and vmdInvSqrt in VML_EP mode detected.\n");
      return 1;
    }
  }

  printf("vdInvSqrt test/example program\n\n");
  printf("           Argument                     vdInvSqrt\n");
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
