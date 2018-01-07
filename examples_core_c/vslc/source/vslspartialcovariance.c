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
!    Calculation of correlation matrix  Example Program Text
!******************************************************************************/

#include <stdio.h>

#include "mkl.h"
#include "errcheck.inc"
#include "statchars.inc"

#define DIM      4            /* Task dimension */
#define PART_DIM (DIM / 2)    /* Partial covariance dimension */

#define EPSILON  1e-6

float cov[DIM][DIM] = {
    {  1.0,  0.1, 0.15,  0.1 },
    {  0.1,  2.0,  0.1,  0.1 },
    { 0.15,  0.1,  1.0,  0.1 },
    {  0.1,  0.1,  0.1,  1.0 }
};

MKL_INT pcov_index[DIM] = { 1, 1, -1, -1 };

int main()
{
    VSLSSTaskPtr task;
    MKL_INT dim;
    MKL_INT pdim;
    MKL_INT cov_storage;
    MKL_INT pcov_storage;
    float cp_cov[DIM][DIM];
    float pcov[PART_DIM][PART_DIM];
    float th_pcov[PART_DIM][PART_DIM];
    int i, i1, j, j1, errcode;
    int errnums = 0;

    /***** Initializing parameters for Summary Statistics task *****/
    dim          = DIM;
    pdim         = PART_DIM;
    cov_storage  = VSL_SS_MATRIX_STORAGE_FULL;
    pcov_storage = VSL_SS_MATRIX_STORAGE_FULL;

    for(i = 0; i < PART_DIM; i++)
    {
        for(j = 0; j < PART_DIM; j++)
        {
            pcov[i][j] = 0.0;
        }
    }

    /***** Create Summary Statistics task *****/
    errcode = vslsSSNewTask( &task, &dim, 0, 0, 0, 0, 0 );
    CheckVslError(errcode);

    /***** Edit task parameters for partial covariance matrix computation *****/
    errcode = vslsSSEditPartialCovCor( task, (int*)pcov_index,
                                       (float*)cov, &cov_storage, 0, 0,
                                       (float*)pcov, &pcov_storage, 0,
                                       &pcov_storage );
    CheckVslError(errcode);

    /***** Compute partial covariance matrix using FAST method *****/
    errcode = vslsSSCompute( task, VSL_SS_PARTIAL_COV, VSL_SS_METHOD_FAST );
    CheckVslError(errcode);

    /***** Printing results *****/
    printf("Task dimension : %d\n\n", dim);

    /* Print input covariance matrix */
    printf(" Covariance matrix\n");
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            printf("%+lf ", cov[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    /* Print computed partial covariance matrix estimate */
    printf(" Computed partial covariance matrix\n");
    for(i = 0; i < pdim; i++)
    {
        for(j = 0; j < pdim; j++)
        {
            printf("%+lf ", pcov[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    /***** Testing stat characteristics of partial covariance matrix *****/
    /* Compute theoretical partial covariance estimate using sweep operator */
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            cp_cov[i][j] = cov[i][j];
        }
    }

    for(i = 0; i < dim; i++)
    {
        if (pcov_index[i] == -1)
            sSweep( i, dim, (float*)cp_cov );
    }

    i1 = 0;
    j1 = 0;
    for(i = 0; i < dim; i++)
    {
        if (pcov_index[i] == 1)
        {
            j1 = 0;
            for(j = 0; j < dim; j++)
            {
                if (pcov_index[j] == 1)
                {
                    th_pcov[i1][j1] = cp_cov[i][j];
                    j1++;
                }
            }
            i1++;
        }
    }

    /* Print theoretical partial covariance estimate */
    printf(" Theoretical partial covariance matrix\n");
    for(i = 0; i < pdim; i++)
    {
        for(j = 0; j < pdim; j++)
        {
            printf("%+lf ", th_pcov[i][j]);
        }
        printf("\n");
    }

    /* Check the correctness of computed partial covariance matrix */
    for(i = 0; i < pdim; i++)
    {
        for(j = 0; j < pdim; j++)
        {
            if(ABS(pcov[i][j] - th_pcov[i][j]) > EPSILON) errnums++;
        }
    }

    /***** Printing summary of the test *****/
    if (errnums == 0)
    {
        printf("\n\nComputed partial covariance matrix estimate is agreed");
        printf(" with theory\n");
    }
    else
    {
        printf("\n\nError: Computed partial covariance matrix estimate");
        printf(" is disagreed with theory\n");
        return 1;
    }

    /***** Delete Summary Statistics task *****/
    errcode = vslSSDeleteTask( &task );
    CheckVslError(errcode);

    MKL_Free_Buffers();

    return 0;
}