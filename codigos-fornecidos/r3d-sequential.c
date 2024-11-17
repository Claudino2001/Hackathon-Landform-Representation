#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"

#define OUTPUT   "output.txt"
#define ANOVA    "anova.txt"

/* Landform Representation Sub-Medio Sao Francisco - Petrolina - Pernambuco (PNZ) */
#define Mxx       5026   
#define Mxy       5026
#define INPUT    "mde.pnz" 

int    n = 0;      // Number of dates in the matrix - global variable
double zm;

void dados   (double *x, double *y, double *z );
void matrizes(double *A, double *B, double *x, double *y , double *z, int N, int r );
void sistema (double *A , double *a, double *B, int N );
void calcula (double *a, double *x, double *y, double *z, int r);
void anova   (double *a, double *x, double *y, double *z, int N, int r );
void pivot   (int l, int m, double *A, double *B , int N );

int main(int argc, char **argv){
 
int r = atoi(argv[1]);        //Degree of a Polynomial
int s = r;                    //Simplify Polynomial Degree
  
int  N   =  (r+1)*(s+1);      //Number of coefficient of polynomial Landform
int  MAX =  (Mxx+1)*(Mxy+1);
   
double *A = (double*) malloc (sizeof(double) * N * N); /*Dynamic Alocation by line - A[i * N + j]  <----------  */
double *B = (double*) malloc (sizeof(double) * N);
double *x = (double*) malloc (sizeof(double) * MAX);
double *y = (double*) malloc (sizeof(double) * MAX);
double *z = (double*) malloc (sizeof(double) * MAX);
double *a = (double*) malloc (sizeof(double) * N);
  
double t1, t2; // Variables for Time measurement using OpenMP

printf("\n[Start...]\n\n");
                   
t1 = omp_get_wtime();
                    
                      printf("\n(Step 1) Extraction of data file mde.pnz \n\n");
                      dados(x, y, z);
   
                      printf("\n(Step 2) Building Matrix \n\n");
                      matrizes(A, B, x, y, z, N, r);
 
                      printf("\n(Step 3) Solver Linear System (Ax=b) \n\n");
                      sistema(A, a, B, N);
                      calcula (a, x, y, z, r);

                      printf("\n(Step 4) Reports Landform \n\n");
                      anova(a, x, y, z, N, r);

t2 = omp_get_wtime();

printf("\n\n[End]\n\n");
   
printf("%d\t%5.2f\n",r, t2-t1);  

free(A);
free(B);
free(a);
free(x);
free(y);
free(z);
  	
return 0;

}/****************************main*************************************************************/

void dados(double *x, double *y, double *z){

     int col, row;
     FILE *f;
     n = 0;

     if ((f = fopen(INPUT,"r")) == NULL)
     {
         printf("\n Erro I/O\n");
         exit(1);
     }
     
     for (row = 0; row < Mxy; ++row)
        for (col = 0; col < Mxx; ++col)
        {
            float h;
            int result = fscanf( f, "%f", &h);
            
            if (h <= 0) continue;
            
            x[n] = row;
            y[n] = col;
            z[n] = h / 2863.0;

            //printf("\n mde[%d,%d] x[%d]=%f  y[%d]=%f z[%d]=%f", row,col, n,x[n], n,y[n], n,z[n] );
            n++;
        }
  
      printf("\n n (number of operations for point in the matrix) = %d\n", n);
          

     fclose(f);

}


void matrizes (double *A, double *B, double *x, double *y, double *z, int N, int r){

 int i, l, c; 
 int s = r;

 for (l = 0; l < N; ++l){
   for (c = 0; c < N; ++c){
      A[l * N + c] = 0.0;

      if (c == 0)
        B[l] = 0.0;

        for (i = 0; i < n; ++i){
            A[l* N + c ] += pow(x[i], (int) (l / (s+1)) + (int) (c / (s+1))) * pow(y[i], l % (r+1) + c % (r+1));
                if (c == 0)
                    B[l] +=  z[i] * pow(x[i], (int) (l / ( s + 1))) * pow(y[i], l % (r + 1));

        }
   }
 }

 printf("\nN (size of the matrix A | number of coefficient of polynomial) = %d\n\n", N);

  for (l = 0; l < N; ++l){
    for (c = 0; c < N; ++c)
       printf("%+1.1e ", A[l*N+c]);
      
    printf( "|%+1.1e\n", B[l]);               
   }

     
}


void pivot (int l, int m, double *A, double *B , int N){

     int    i, l_maior;
     double maior, auxA,auxB;

     maior   = A[l*N+l];
     l_maior = l;

     for (i = l; i < m; ++i)
        if( fabs(maior) < fabs(A[i*N+l]) )
        {
            maior   = A[i*N+l];
            l_maior = i;
        }

     if (l != l_maior)
     {
        for (i = l; i < m; ++i)
        {
           auxA           = A[l*N+i];
           A[l*N+i]       = A[l_maior*N+i];
           A[l_maior*N+i] = auxA;
        }

        auxB       = B[l];
        B[l]       = B[l_maior];
        B[l_maior] = auxB;
     }

}

void sistema (double *A, double *a, double *B, int N)
{
     int    i, j, l, c;
     double aux;
     
     printf( " ---------------------------------\n");
     for (l = 0; l < N; ++l)
     {
        for (c = 0; c < N; ++c)
           printf(" %+1.1e", A[l*N+c] );
        
      printf("|%+1.1e\n", B[l] );               
     }
     
     for (i = 0; i < N; ++i)
     {
        pivot(i, N, A, B, N);
        aux = A[i*N+i];
        for( j = 0; j < N; ++j )
           if( aux != 0.0 ) A[i*N+j] /= aux;
        if(aux != 0.0) B[i] /= aux;
        for( l = i+1; l < N; ++l )
        {
           aux = A[l*N+i];
           for( c = i; c < N; ++c )
               A[l*N+c] -= aux * A[i*N+c];
           B[l] -= aux * B[i];
        }
        
        printf( " ---------------------------------\n");
        for (l = 0; l < N; ++l)
        {
           for(c = 0; c < N; ++c)
           {
              printf("%+1.1e ", A[l*N+c] );
           }
           printf("|%+1.1e\n", B[l] );               
        }
     }
     a[N-1] = B[N-1];
     printf( "a[%d]= %12g\n",N-1,a[N-1] );     
     for (l = N-2; l >= 0; --l)
     {
        a[l] = B[l];
        for (c = l+1; c < N; ++c)
            a[l] -= A[l*N+c] * a[c];
        printf( "a[%d]= %12g\n", l, a[l] );      
     }

}


void calcula (double *a, double *x, double *y, double *z, int r)
{
     int    i, c, l;
     double Sx, Sy, Sz, e, ze, Se, Sze;
     FILE   *f;
     int s = r;

     Sx= Sy= Sz= Se= Sze= 0.0;

     if (( f = fopen(OUTPUT,"w")) == NULL)
     {
        printf("\n Erro I/O");
        exit(3);
     }

     fprintf(f," ==========================================================\n");
     fprintf(f,"     i      x[i]         y[i]        ye[i]         e[i]   \n");
     fprintf(f," ==========================================================\n");

     for (i = 0; i < n; i++)
     {
        Sx  += x[i];
        Sz  += z[i];

        ze = 0.0;

        for (c = 0; c < r+1; c++)
           for (l = 0; l < s+1; l++)
              ze += a[c*(r+1)+l] * pow(x[i],c) * pow(y[i],l);

        e    = z[i] - ze;
        Se  += e;
        Sze += ze;

        fprintf(f," %5d    %9.5e    %9.5e    %9.5e    %9.5e\n", i, x[i],y[i], ze, e);
     }

     fprintf(f," ============================================================\n");
     fprintf(f," %5d    %9.5e   %9.5e   %9.5e    %9.5e\n", n, Sx, Sy, Sze, Se);

     zm = Sz / n;

     fclose(f);

}

void anova (double *a, double *x, double *y, double *z, int N, int r)
{
     int    i, glReg, glR, glT, c, l;
     double SQReg, SQR, SQT, QMReg, QMR, R2, F, ze;
     FILE   *f;
     int s=r;

     SQR = SQReg = 0.0;

     glReg = N;
     glR   = n - 2*N;
     glT   = n - N;

     printf("\n--------------------------------------------\n");
     for(c = 0; c < r+1; ++c){
         for(l = 0; l < s+1; ++l)
           printf("a[%2d,%2d]= %2d\n",c,l, c*(r+1)+l);
     }
     printf("\n---------------------------------------------\n\n");
    
     
     for (i = 0; i < n; ++i)
     {
        ze = 0.0;
        for (c = 0; c < r+1; ++c)
           for (l = 0; l < s+1; ++l)
              ze += a[c*(r+1)+l] * pow(x[i], c) * pow(y[i], l);

        SQReg += (ze   - zm) * (ze   - zm);
        SQR   += (z[i] - ze) * (z[i] - ze);
     }

     SQT   = SQReg + SQR;
     QMReg = SQReg / glReg;
     QMR   = SQR   / glR;
     F     = QMReg / QMR;
     R2    = SQReg / SQT;

     if ((f=fopen(ANOVA,"w")) == NULL )
     {
        printf("\n Erro I/O\n");
        exit(2);
     }

     fprintf(f," \n\n\n\n");
     fprintf(f," ANOVA\n");
     fprintf(f," =================================================\n");
     fprintf(f," FV           gl      SQ         QM          F    \n");
     fprintf(f," =================================================\n");
     fprintf(f," Regression  %5d  %12e  %12e  %12e\n", glReg, SQReg, QMReg, F);
     fprintf(f," Residue     %5d  %12e  %12e      \n", glR  , SQR  , QMR     );
     fprintf(f," -------------------------------------------------\n");
     fprintf(f," Total       %5d  %12e            \n", glT  , SQT            );
     fprintf(f," =================================================\n");
     fprintf(f," R^2= %12e                        \n", R2       );

     printf ("P(x)="); 
   
     for (c = 0; c < r+1; ++c)
        for (l = 0; l < s+1; ++l){
           fprintf(f,"+x^%d*y^%d*\t%12g\n", c, l, a[c*(r+1)+l]);
           printf ("%12g*(x**%d)*(y**%d) + ",a[c*(r+1)+l], c, l ); 
      }
       
     fclose(f);

}