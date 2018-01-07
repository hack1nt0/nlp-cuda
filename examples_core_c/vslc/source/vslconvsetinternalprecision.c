/*******************************************************************************
* Copyright 2003-2015 Intel Corporation All Rights Reserved.
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
!    vslConvSetInternalPrecision  Example Program Text
!******************************************************************************/

#include "mkl_vsl.h"

#include <stdio.h>

int main()
{
    VSLConvTaskPtr task;
    int status,ok;
    MKL_INT mode,rank,xshape,yshape,zshape,precision;

    ok = 1;
    printf("EXAMPLE re-setting internal precision\n");

    mode = VSL_CONV_MODE_AUTO;
    rank = 1;
    xshape = 100;
    yshape = 1000;
    zshape = (xshape-1) + (yshape-1) + 1;
    vslsConvNewTask(&task,mode,rank,&xshape,&yshape,&zshape);

    precision = VSL_CONV_PRECISION_DOUBLE;
    status = vslConvSetInternalPrecision(task,precision);

    if (status != VSL_STATUS_OK) {
        printf("ERROR: bad status: %d\n",status);
        ok = 0;
    }

    printf("EXAMPLE %s\n", ok? "PASSED": "FAILED");
    return !ok;
}
