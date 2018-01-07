/*******************************************************************************
* Copyright 1999-2015 Intel Corporation All Rights Reserved.
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
!      mklservicefunctions example program demonstrates a lot of
!      MKL service functions.
!******************************************************************************/
#include <stdio.h>
#include <mkl.h>

#define NN 1000
#define buf_len 198

#if defined(MKL_ILP64) && defined(_WIN32)
#define FORMAT "%I64d"
#else
#define FORMAT "%lld"
#endif 

static void aux_print_n_threads(void);
static void aux_call_dgemm(MKL_INT N, double* A, double* B, double* C, int PRINT_CLOCKS);

int main() {

    double freq, seconds_s, seconds_e;

    double *a;
    double *b;
    double *c;

    int n = NN;

    MKL_INT64 allocated_bytes;
    int       allocated_buffers;

    char buf[buf_len];
    MKLVersion ver;

/** Information and defaults **/

    seconds_s = dsecnd();

    printf("\nMKL service functions example started\n");

    printf("\nMKL release version:\n");
    MKL_Get_Version_String(buf, buf_len);
    printf("%s\n",buf);

    MKL_Get_Version(&ver);
    printf("    Major version:          %d\n",ver.MajorVersion);
    printf("    Minor version:          %d\n",ver.MinorVersion);
    printf("    Update version:         %d\n",ver.UpdateVersion);
    printf("    Product status:         %s\n",ver.ProductStatus);
    printf("    Build:                  %s\n",ver.Build);
    printf("    Platform:               %s\n",ver.Platform);
    printf("    Processor optimization: %s\n",ver.Processor);

    printf("\nInformation and defaults\n");

    freq = MKL_Get_Cpu_Frequency();
    printf("    Current CPU frequency:%8.4fGHz\n",freq);

    freq = MKL_Get_Max_Cpu_Frequency();
    printf("    Maximum CPU frequency:%8.4fGHz\n",freq);

    freq = MKL_Get_Clocks_Frequency();
    printf("    Frequency:%8.4fGHz\n",freq);

    aux_print_n_threads();

/** Memory functions **/
    MKL_Peak_Mem_Usage(MKL_PEAK_MEM_ENABLE);
    printf("\nMemory functions\n");
    printf("    Allocate DGEMM's arrays\n");

    a = (double*)MKL_malloc(n*n*sizeof(double),128);
    b = (double*)MKL_malloc(n*sizeof(double),128); /** Allocates a memory buffer of smaller size to realloc it to appropriate size later **/
    c = (double*)MKL_calloc(n*n,sizeof(double),128);

    if ( a == NULL || b == NULL || c == NULL ) {
        printf("\nAllocation of arrays failed\n");
        return 1;
    } else {
        b = (double*)MKL_realloc(b,n*n*sizeof(double));
        printf("    CALL DGEMM\n");
        aux_call_dgemm(n,a,b,c,0);
        printf("    ...Done\n");

        allocated_bytes = MKL_Peak_Mem_Usage(MKL_PEAK_MEM);
        printf("    Peak memory allocated by MKL allocator :"FORMAT" bytes.\n",allocated_bytes);

        allocated_bytes = MKL_Mem_Stat(&allocated_buffers);
        printf("    Currently allocated by MKL allocator :"FORMAT" bytes in%3d buffers.\n",allocated_bytes,(int)allocated_buffers);

        MKL_Free_Buffers();
        allocated_bytes = MKL_Mem_Stat(&allocated_buffers);
        printf("    After Mkl_Free_Buffers was called:\n");
        printf("    Currently allocated by MKL allocator :"FORMAT" bytes in%3d buffers.\n",allocated_bytes,(int)allocated_buffers);
        if ( MKL_Peak_Mem_Usage(MKL_PEAK_MEM_DISABLE) < 0 ) {
            printf("Peak memory statistics is not disabled\n");
            return 1;
        }
/** Threading functions **/

        printf("\nDGEMM & MKL threading\n");

        aux_call_dgemm(n,a,b,c,1); /** DGEMM on N threads (default) **/
        (void) MKL_Domain_Set_Num_Threads(1,MKL_DOMAIN_BLAS);
        aux_call_dgemm(n,a,b,c,1); /** DGEMM on 1 thread **/

/** DGEMM on XXX threads. MKL_DYNAMIC **/

        printf("\nMKL_DYNAMIC experiment\n");

        printf("    Force MKL_DYNAMIC=TRUE\n");
        MKL_Set_Dynamic(1);
        printf("        Set MKL BLAS-N-Threads to 64\n");
        (void) MKL_Domain_Set_Num_Threads(64,MKL_DOMAIN_BLAS);
        aux_print_n_threads();
        aux_call_dgemm(n,a,b,c,1);

        printf("\n    Switch off MKL_DYNAMIC facility\n");
        MKL_Set_Dynamic(0);
        printf("        Set MKL BLAS-N-Threads to 64\n");
        (void) MKL_Domain_Set_Num_Threads(64,MKL_DOMAIN_BLAS);
        aux_print_n_threads();
        aux_call_dgemm(n,a,b,c,1);

        printf("\n    Free DGEMM's arrays\n");
        MKL_free(a);
        MKL_free(b);
        MKL_free(c);
    }  
    seconds_e = dsecnd()-seconds_s;
    printf("\nMKL service functions example finished at%8.4f seconds\n",seconds_e);

    return 0;
}

static void aux_call_dgemm(MKL_INT N, double* A, double* B, double* C, int PRINT_CLOCKS) {

    int      i, j;
    char     transa, transb;
    double   alpha, beta;

    unsigned MKL_INT64  dgemm_s, dgemm_e;
    unsigned MKL_INT64  dgemm_clocks, dgemm_cb, dgemm_ct, dgemm_cu;

    transa = 'N'; transb = 'N';
    alpha = 1.1;  beta = -1.2;

    for ( i=0; i<N; i++ ) {
        for ( j=0; j<N; j++ ) {
            A[j*N+i] = (double)(i+1);
            B[j*N+i] = (double)(-(i+1));
        }
    }

    MKL_Get_Cpu_Clocks(&dgemm_s);
    dgemm(&transa,&transb,&N,&N,&N,&alpha,A,&N,B,&N,&beta,C,&N);
    mkl_get_cpu_clocks(&dgemm_e);

    dgemm_clocks = dgemm_e-dgemm_s;
    if ( PRINT_CLOCKS ) {
        dgemm_cb = dgemm_clocks/1000000;
        dgemm_ct = (dgemm_clocks-(dgemm_cb*1000000))/1000;
        dgemm_cu = (dgemm_clocks-(dgemm_cb*1000000)-(dgemm_ct*1000));
        printf("    DGEMM (%4d) on %2d thread(s):%08.3d.%03d.%03d clocks\n",
               (int)N,MKL_Domain_Get_Max_Threads(MKL_DOMAIN_BLAS),(int)dgemm_cb,(int)dgemm_ct,(int)dgemm_cu);
    }
}

static void aux_print_n_threads(void) {

    if ( MKL_Get_Dynamic() == 0 ) {
        printf("\n        MKL_DYNAMIC : FALSE\n");
    } else {
        printf("\n        MKL_DYNAMIC : TRUE\n");
    }

    printf("        MKL Number of THREADS:  ALL  BLAS FFT PARDISO VML\n");
    printf("                              %4d %4d %4d %4d %4d\n",
           MKL_Get_Max_Threads(),
           MKL_Domain_Get_Max_Threads(MKL_DOMAIN_BLAS),
           MKL_Domain_Get_Max_Threads(MKL_DOMAIN_FFT),
           MKL_Domain_Get_Max_Threads(MKL_DOMAIN_PARDISO),
           MKL_Domain_Get_Max_Threads(MKL_DOMAIN_VML));

    return;
}