import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule

def grkf45_1d_module():
    source_module = SourceModule("""

    #include <cstdlib>
    #include <cmath>
    #include <ctime>
    # define MAXNFE 3000
    
    using namespace std;

    /* USER GIVEN DEVICE FUNCTION: parameter = a */
    __device__ void r4_f0 ( float t, float y0, float *yp0, float a){

    *yp0 = 1.0 + y0*y0 + a*sin(t);

    /* printf("(*_*)< t=%2.8f,y0= %2.8f,yp0= %2.8f \\n",t,y0,yp0); */

    return;
    }




    /* DEVICE FUNCTION */

    __device__ void r4_fehl (float y0, float t, float h, float yp0, float *f1_0, float *f2_0, float *f3_0, float *f4_0, float *f5_0, float a){ 
    
    float ch;
    int i;
    float s0;

    ch = h / 4.0;

    /* for ( i = 0; i < neqn; i++ ){ */
    *f5_0 = y0 + ch * yp0;
    /* } */

    r4_f0 ( t + ch, *f5_0, f1_0, a); 

    ch = 3.0 * h / 32.0;

    /* for ( i = 0; i < neqn; i++ ){ */
    *f5_0 = y0 + ch * ( yp0 + 3.0 * *f1_0 );
    /* } */
    
    r4_f0 ( t + 3.0 * h / 8.0, *f5_0, f2_0, a);
    ch = h / 2197.0;
    
    /* for ( i = 0; i < neqn; i++ ){ */
    *f5_0 = y0 + ch * 
    ( 1932.0 * yp0
    + ( 7296.0 * *f2_0 - 7200.0 * *f1_0 ) 
    );
    /* } */

    r4_f0 ( t + 12.0 * h / 13.0, *f5_0, f3_0, a);
    ch = h / 4104.0;

    /* for ( i = 0; i < neqn; i++ ){ */
    *f5_0 = y0 + ch * 
    ( 
    ( 8341.0 * yp0 - 845.0 * *f3_0 ) 
    + ( 29440.0 * *f2_0 - 32832.0 * *f1_0 ) 
    );
    /* } */
    
    r4_f0 ( t + h, *f5_0, f4_0, a);
    ch = h / 20520.0;
    
    /* for ( i = 0; i < neqn; i++ ){ */
    *f1_0 = y0 + ch * 
    ( 
    ( -6080.0 * yp0 
    + ( 9295.0 * *f3_0 - 5643.0 * *f4_0 ) 
    ) 
    + ( 41040.0 * *f1_0 - 28352.0 * *f2_0 ) 
    );
    /* } */
    
    r4_f0 ( t + h / 2.0, *f1_0, f5_0, a);
    ch = h / 7618050.0;
    
    /* for ( i = 0; i < neqn; i++ ){ */
    s0 = y0 + ch * 
    ( 
    ( 902880.0 * yp0 
    + ( 3855735.0 * *f3_0 - 1371249.0 * *f4_0 ) ) 
    + ( 3953664.0 * *f2_0 + 277020.0 * *f5_0 ) 
    );
    /* } */

    *f1_0 = s0;

    /* printf("(*_*)< +++++ %2.8f,%2.8f,%2.8f,%2.8f, %2.8f \\n",f1_0,f2_0,f3_0,f4_0,f5_0); */

    return;
    }


    /* GLOBAL FUNCTION */
    __global__ void r4_rkf45_1d (int* flagM, float* aM, float *yM, float tM, float toutM, float relerr, float abserr){

    float ae;
    float dt;
    float ee;
    float eeoet;
    const float eps = 1.19209290E-07;
    float esttol;
    float et;
    float f1_0;
    float f2_0;
    float f3_0;
    float f4_0;
    float f5_0;
    float h = -1.0;
    bool hfaild;
    float hmin;
    int i;
    int init = -1000;
    int k;
    int kop = -1;
    int nfe = -1;
    float s;
    float scale;
    float tol;
    float toln;
    float ypk;
    bool output;

    /* user defined parameters */
    float a;
    int ib = blockIdx.x;

    a = aM[ib];


    /* USE register */
    float t;
    float y0;
    float yp0;
    float tout;

    t = tM;
    tout = toutM;
    y0 = yM[ib];
    r4_f0 ( t, y0, &yp0, a);
    
    dt = tout - t;

    if ( init == 0 ){

    init = 1;
    h = abs( dt );
    toln = 0.0;

    /* for ( k = 0; k < neqn; k++ ){ */
    tol = (relerr) * abs( y0 ) + abserr;

    if ( 0.0 < tol ){
    toln = tol;
    ypk = abs( yp0 );
    if ( tol < ypk * pow ( h, 5 ) )
    {
    h = ( float ) pow ( ( double ) ( tol / ypk ), 0.2 );
    }}

    /* } */
    

    if ( toln <= 0.0 ){h = 0.0;}
    h = max ( h, 26.0 * eps * max ( abs ( t ), abs ( dt ) ) );
    }


    /* SIGN(positive/negative -> 1/-1) to signbit(positive/negative -> 0/1) in CUDA math API */

    h = ( - 2.0* signbit( dt ) + 1.0 ) *abs( h );

    if ( 2.0 * abs( dt ) <= abs( h ) ){
    kop = kop + 1;
    }

    output = false;
    scale = 2.0 / (relerr);
    ae = scale * abserr;

    for ( ; ; ){
    hfaild = false;
    hmin = 26.0 * eps * abs ( t );
    dt = tout - t;

    if ( 2.0 * abs ( h ) <= abs ( dt ) ){
    }else{
    if ( abs ( dt ) <= abs ( h ) ){
    output = true;
    h = dt;
    }else{
    h = 0.5 * dt;
    }
    }

    for ( ; ; ){

    if ( MAXNFE < nfe ){
    
    tM = t;
    yM[ib] = y0;
    flagM[ib]=4;

    printf("(*_*)< t=%2.8f \\n",t);
    printf("*WARNING! END MAXNFE < nfe condition! \\n");

    return;
    }

    /* printf("(*_*)< >>>>> %2.8f,%2.8f,%2.8f,%2.8f, %2.8f \\n",f1_0,f2_0,f3_0,f4_0,f5_0); */
    r4_fehl (y0, t, h, yp0, &f1_0, &f2_0, &f3_0, &f4_0, &f5_0, a);

    /* printf("(*_*)< <<<<< %2.8f,%2.8f,%2.8f,%2.8f, %2.8f \\n",f1_0,f2_0,f3_0,f4_0,f5_0); */

    nfe = nfe + 5;
    eeoet = 0.0;

 
    /* for ( k = 0; k < neqn; k++ ){ */
    et = abs ( y0 ) + abs ( f1_0 ) + ae;
    
    ee = abs ( ( -2090.0 * yp0 
    + ( 21970.0 * f3_0 - 15048.0 * f4_0 ) 
    ) 
    + ( 22528.0 * f2_0 - 27360.0 * f5_0 ) 
    );
    
    eeoet = max ( eeoet, ee / et );

    /* } */

    esttol = abs ( h ) * eeoet * scale / 752400.0;

    if ( esttol <= 1.0 )
    {
    break;
    }

    hfaild = true;
    output = false;

    /* printf("(*_*)< h = %2.8f, esttol= %2.8f \\n",h, esttol); */
    
    if ( esttol < 59049.0 ){
    s = 0.9 / ( float ) pow ( ( double ) esttol, 0.2 );
    }else{
    s = 0.1;
    }

    h = s * h;
    }

    t = t + h;
    /* for ( i = 0; i < neqn; i++ ){ */
      y0 = f1_0;
    /* } */

    r4_f0 ( t, y0, &yp0, a);

    nfe = nfe + 1;
    if ( 0.0001889568 < esttol )
    {
    s = 0.9 / ( float ) pow ( ( double ) esttol, 0.2 );
    }
    else
    {
    s = 5.0;
    }

    if ( hfaild )
    {
    s = min ( s, 1.0 );
    }

    h = ( - 2.0* signbit( h ) + 1.0 ) * max ( s * abs ( h ), hmin );

    if (output){

    tM = t;
    yM[ib] = y0;
    flagM[ib]=2;

    /* printf("Normal Exit N=%d\\n",nfe); */

    return;
    }
    
    }


    return;

    }

    """,options=['-use_fast_math'])

    return source_module


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("*******************************************")
    print("GPU RKF45 1D ODE solver for the following example.")
    print("y'=1+y*y+a sin(t), y[0]=0 on t=[0,1.4] ")
    print("*******************************************")

    source_module=grkf45_1d_module()
    pkernel=source_module.get_function("r4_rkf45_1d")

    eps=1.19209290e-07
    relerr=np.sqrt(eps)
    abserr=np.sqrt(eps)
    print("abserr=",abserr)
    print("relerr=",relerr)


    nw=1
    nt=16
    nq=1
    nb = nw*nt*nq 
    sharedsize=0 #byte

    #parameters
    a=np.linspace(-0.1,0.1,nb)
    a=a.astype(np.float32)
    dev_a = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(dev_a,a)
    
    y0=np.zeros(nb)
    y0=y0.astype(np.float32)
    dev_y0 = cuda.mem_alloc(y0.nbytes)
    cuda.memcpy_htod(dev_y0,y0)
    
    flag=np.zeros(nb)
    flag=flag.astype(np.int32)
    dev_flag = cuda.mem_alloc(flag.nbytes)
    cuda.memcpy_htod(dev_flag,flag)


    t=np.linspace(0.0,1.4,25)
    yarr=[]
    yarr.append(np.copy(y0))            
    for j,tnow in enumerate(t[:-1]):
        tin=tnow
        tout=t[j+1]
        pkernel(dev_flag, dev_a,dev_y0, np.float32(tin),np.float32(tout),np.float32(relerr),np.float32(abserr),block=(int(nw),1,1), grid=(int(nt),int(nq)),shared=sharedsize)

        cuda.memcpy_dtoh(y0, dev_y0)
        cuda.memcpy_dtoh(flag, dev_flag)
        yarr.append(np.copy(y0))

    yarr=np.array(yarr)

    plt.plot(t,yarr,".",color="C0")
    plt.plot(t,yarr,alpha=0.3)
    plt.show()
    
