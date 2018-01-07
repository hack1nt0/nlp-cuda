/*******************************************************************************
* Copyright 2010-2015 Intel Corporation All Rights Reserved.
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

#include <stdlib.h>
#include <math.h>

#define _DF_SINGLE_PI 3.141592653589793115997963468544185161590576171875
#define _DF_DOUBLE_PI 3.141592653589793115997963468544185161590576171875

/*
// "Bubble" sort for single precision data
*/
static int sSort( int n, float* a )
{
    int i, j;
    float tmp;

    for ( i = 0; i < n-1; i++ )
    {
        for ( j = i+1; j < n; j++ )
        {
            if (a[j] < a[i])
            {
                tmp = a[j];
                a[j] = a[i];
                a[i] = tmp;
            }
        }
    }

    return 0;
}

/*
// "Bubble" sort for double precision data
*/
static int dSort( int n, double* a )
{
    int i, j;
    double tmp;

    for ( i = 0; i < n-1; i++ )
    {
        for ( j = i+1; j < n; j++ )
        {
            if (a[j] < a[i])
            {
                tmp = a[j];
                a[j] = a[i];
                a[i] = tmp;
            }
        }
    }

    return 0;
}

/*
// Generate single precision array of functions with following
// dependency: f(x) = Sin( 2 * Pi * freq * x ),
// the grid is not uniform
*/
static int sSinDataNotUniformGrid(float* f, float* x, float freq, int n)
{
    int i;

    for( i = 0; i < n; i++ )
    {
        f[i] = sinf( 2.0 * _DF_SINGLE_PI * freq * x[i] );
    }

    return 0;
}

/*
// Generate double precision array of dependent variables with following
// dependency: f(x) = Sin( 2 * Pi * freq * x ),
// the grid is not uniform
*/
static int dSinDataNotUniformGrid(double* f, double* x, double freq, int n)
{
    int i;

    for( i = 0; i < n; i++ )
    {
        f[i] = sin( 2.0 * _DF_DOUBLE_PI * freq * x[i] );
    }

    return 0;
}

/*
// Generate single precision array of dependent variables with following
// dependency: f(x) = 2 * Pi * freq * Cos( 2 * Pi * freq * x ),
// the grid is not uniform
*/
static int sSinDerDataNotUniformGrid(float* f, float* x, float freq, int n)
{
    int i;
    float mult;

    mult = 2.0 * _DF_SINGLE_PI * freq;

    for( i = 0; i < n; i++ )
    {
        f[i] = mult * cosf( mult * x[i] );
    }

    return 0;
}

/*
// Generate double precision array of dependent variables with following
// dependency: f(x) = 2 * Pi * freq * Cos( 2 * Pi * freq * x ),
// the grid is not uniform
*/
static int dSinDerDataNotUniformGrid(double* f, double* x, double freq, int n)
{
    int i;
    double mult;

    mult = 2.0 * _DF_SINGLE_PI * freq;

    for( i = 0; i < n; i++ )
    {
        f[i] = mult * cos( mult * x[i] );
    }

    return 0;
}

/*
// Generate single precision array of dependent variables with following
// dependency: f(x) = -4 * Pi * Pi * freq * freq * Sin( 2 * Pi * freq * x ),
// the grid is not uniform
*/
static int sSinDer2DataNotUniformGrid(float* f, float* x, float freq, int n)
{
    int i;
    float mult;

    mult = 2.0 * _DF_SINGLE_PI * freq;

    for( i = 0; i < n; i++ )
    {
        f[i] = -mult * mult * sinf( mult * x[i] );
    }

    return 0;
}

/*
// Generate double precision array of dependent variables with following
// dependency: f(x) = -4 * Pi * Pi * freq * freq * Sin( 2 * Pi * freq * x ),
// the grid is not uniform
*/
static int dSinDer2DataNotUniformGrid(double* f, double* x, double freq, int n)
{
    int i;
    double mult;

    mult = 2.0 * _DF_SINGLE_PI * freq;

    for( i = 0; i < n; i++ )
    {
        f[i] = -mult * mult * sin( mult * x[i] );
    }

    return 0;
}

/*
// Generate single precision array of dependent variables with following
// dependency: f(x) = Sin( 2 * Pi * freq * x ),
// the grid is uniform
*/
static int sSinDataUniformGrid(float* f, float freq,
                               float left, float right, int n)
{
    int i;
    float delta, cur_x;

    delta = (right - left) / ((float)n - 1);

    cur_x = left;
    for( i = 0; i < n; i++ )
    {
        f[i] = sinf( 2.0 * _DF_SINGLE_PI * freq * cur_x );
        cur_x += delta;
    }

    return 0;
}

/*
// Generate double precision array of dependent variables with following
// dependency: f(x) = Sin( 2 * Pi * freq * x ),
// the grid is uniform
*/
static int dSinDataUniformGrid(double* f, double freq,
                               double left, double right, int n)
{
    int i;
    double delta, cur_x;

    delta = (right - left) / ((double)n - 1);

    cur_x = left;
    for( i = 0; i < n; i++ )
    {
        f[i] = sin( 2.0 * _DF_SINGLE_PI * freq * cur_x );
        cur_x += delta;
    }

    return 0;
}

/*
// Generate single precision array of independent variables with uniform grid
*/
static int sUniformData(float* x, float left, float right, int n)
{
    int i;
    float delta;

    delta = (right - left) / ((float)n - 1);

    for( i = 0; i < n; i++ )
    {
        x[i] = left + (float)i * delta;
    }

    return 0;
}

/*
// Generate double precision array of independent variables with uniform grid
*/
static int dUniformData(double* x, double left, double right, int n)
{
    int i;
    double delta;

    delta = (right - left) / ((double)n - 1);

    for( i = 0; i < n; i++ )
    {
        x[i] = left + (double)i * delta;
    }

    return 0;
}

/*
// Generate single precision array of independent variables
// with quasi-uniform grid
*/
static int sQuasiUniformData(float* x, float left, float right, int n)
{
    int errcode = 0;
    float eps;

    errcode = sUniformData(x, left, right, n);

    eps = ( x[0] - x[1] ) / 5.0;

    x[1] += eps;

    return 0;
}

/*
// Generate double precision array of independent variables
// with quasi-uniform grid
*/
static int dQuasiUniformData(double* x, double left, double right, int n)
{
    int errcode = 0;
    double eps;

    errcode = dUniformData(x, left, right, n);

    eps = ( x[0] - x[1] ) / 5.0;

    x[1] += eps;

    return 0;
}

/*
// Generate single precision array of uniformly distributed variables
*/
static int sUniformRandData(float* x, float left, float right, int n)
{
    VSLStreamStatePtr stream;
    MKL_INT brng = VSL_BRNG_WH;
    MKL_INT seed = 7777777;
    MKL_INT method = 0;
    int errcode;

    /***** Initialize *****/
    errcode = vslNewStream( &stream, brng, seed );
    if ( errcode < 0 ) return errcode;

    /***** Call RNG *****/
    errcode = vsRngUniform( method, stream, n, x, left, right );
    if ( errcode < 0 ) return errcode;

    /***** Deinitialize *****/
    errcode = vslDeleteStream( &stream );

    return errcode;
}

/*
// Generate double precision array of uniformly distributed variables
*/
static int dUniformRandData(double* x, double left, double right, int n)
{
    VSLStreamStatePtr stream;
    MKL_INT brng = VSL_BRNG_WH;
    MKL_INT seed = 7777777;
    MKL_INT method = VSL_RNG_METHOD_UNIFORM_STD;
    int errcode;

    /***** Initialize *****/
    errcode = vslNewStream( &stream, brng, seed );
    if ( errcode < 0 ) return errcode;

    /***** Call RNG *****/
    errcode = vdRngUniform( method, stream, n, x, left, right );
    if ( errcode < 0 ) return errcode;

    /***** Deinitialize *****/
    errcode = vslDeleteStream( &stream );

    return errcode;
}

/*
// Generate single precision sorted array of uniformly distributed variables
*/
static int sUniformRandSortedData(float* x, float left, float right, int n)
{
    int errcode = 0;

    errcode = sUniformRandData(x, left, right, n);
    if ( errcode < 0 ) return errcode;

    sSort(n, x);

    return errcode;
}

/*
// Generate double precision sorted array of uniformly distributed variables
*/
static int dUniformRandSortedData(double* x, double left, double right, int n)
{
    int errcode = 0;

    errcode = dUniformRandData(x, left, right, n);
    if ( errcode < 0 ) return errcode;

    dSort(n, x);

    return errcode;
}

static int sPeakData(float* x, float a, float sigma, int n)
{
    VSLStreamStatePtr stream;
    MKL_INT brng = VSL_BRNG_WH;
    MKL_INT seed = 7777777;
    MKL_INT method = VSL_RNG_METHOD_GAUSSIAN_ICDF;
    int errcode;

    /***** Initialize *****/
    errcode = vslNewStream( &stream, brng, seed );
    if ( errcode < 0 ) return errcode;

    /***** Call RNG *****/
    errcode = vsRngGaussian( method, stream, n, x, a, sigma );
    if ( errcode < 0 ) return errcode;

    /***** Deinitialize *****/
    errcode = vslDeleteStream( &stream );

    return errcode;
}

static int dPeakData(double* x, double a, double sigma, int n)
{
    VSLStreamStatePtr stream;
    MKL_INT brng = VSL_BRNG_WH;
    MKL_INT seed = 7777777;
    MKL_INT method = VSL_RNG_METHOD_GAUSSIAN_ICDF;
    int errcode;

    /***** Initialize *****/
    errcode = vslNewStream( &stream, brng, seed );
    if ( errcode < 0 ) return errcode;

    /***** Call RNG *****/
    errcode = vdRngGaussian( method, stream, n, x, a, sigma );
    if ( errcode < 0 ) return errcode;

    /***** Deinitialize *****/
    errcode = vslDeleteStream( &stream );
    return errcode;
}