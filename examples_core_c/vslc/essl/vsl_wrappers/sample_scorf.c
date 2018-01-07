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

#include "mkl_vsl.h"

#include <stdio.h>
#include <stdlib.h>

void scorf(
    int init,
    float h[], int inc1h,
    float x[], int inc1x, int inc2x,
    float y[], int inc1y, int inc2y,
    int nh, int nx, int m, int iy0, int ny,
    void* aux1, int naux1, void* aux2, int naux2)
{
    int status = VSL_STATUS_OK ,error,i;
    VSLCorrTaskPtr task;

    if (init != 0)
        return; /* ignore aux1, aux2 */

    vslsCorrNewTaskX1D(&task,
        VSL_CORR_MODE_FFT,nh,nx,ny,
        h,inc1h);
    vslCorrSetStart(task, &iy0);

    /* task is implicitly committed at i==0 */
    for (i=0; i<m; i++) {
        float* xi = &x[inc2x * i];
        float* yi = &y[inc2y * i];
        status = vslsCorrExecX1D(task,
            xi,inc1x, yi,inc1y);
        /* check status later */
    }

    error = vslCorrDeleteTask(&task);

    if (status != VSL_STATUS_OK) {
        printf("ERROR: scorf(): bad status=%d\n",status);
        exit(1);
    }
    if (error != 0) {
        printf("ERROR: scorf(): failed to destroy the task descriptor\n");
        exit(1);
    }
}