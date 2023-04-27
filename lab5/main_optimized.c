#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_blas.h>
#include <sys/times.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/wait.h>

#define START_SIZE 100
#define MAX_SIZE 1000
#define STEP 50

struct timespec start, end;

double get_time()
{
    return (double)clock() / CLOCKS_PER_SEC;
}
void generate_line(int size, char *type, double time)
{
    FILE *report = fopen("c2_results.csv", "a");
    fprintf(report, "%d,%s,%f\n", size, type, time);
    fclose(report);
}

void naive_multiplication(double **A, double **B, double **C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            for (int k = 0; k < size; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void better_multiplication(double **A, double **B, double **C, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < size; k++)
        {
            for (int j = 0; j < size; j++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void blas_multiplication(double *a, double *b, double *c, int rows)
{
    gsl_matrix_view D = gsl_matrix_view_array(a, rows, rows);
    gsl_matrix_view E = gsl_matrix_view_array(b, rows, rows);
    gsl_matrix_view F = gsl_matrix_view_array(c, rows, rows);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                   1.0, &D.matrix, &E.matrix,
                   0.0, &F.matrix);
}

void declare_matrix(double ***A, double ***B, double ***C, double **a, double **b, double **c, int size)
{
    (*A) = calloc(size, sizeof(double *));
    (*B) = calloc(size, sizeof(double *));
    (*C) = calloc(size, sizeof(double *));
    (*a) = calloc(size * size, sizeof(double));
    (*b) = calloc(size * size, sizeof(double));
    (*c) = calloc(size * size, sizeof(double));
    for (int i = 0; i < size; i++)
    {
        (*A)[i] = calloc(size, sizeof(double));
        (*B)[i] = calloc(size, sizeof(double));
        (*C)[i] = calloc(size, sizeof(double));
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            (*A)[i][j] = rand() % 10;
            (*A)[i][j] = rand() % 10;
        }
    }
}

void free_matrix(double ***A, double ***B, double ***C, double **a, double **b, double **c, int size)
{
    for (int i = 0; i < size; i++)
    {
        free((*A)[i]);
        free((*B)[i]);
        free((*C)[i]);
    }
    free((*A));
    free((*B));
    free((*C));
    free((*a));
    free((*b));
    free((*c));
}

void test()
{
    double **A, **B, **C;
    double *a, *b, *c;
    double time1, time2, time3;
    for (int i = START_SIZE; i < MAX_SIZE; i += STEP)
    {
        declare_matrix(&A, &B, &C, &a, &b, &c, i);
        double start, end;
        if (fork() == 0)
        {
            start = get_time();
            naive_multiplication(A, B, C, i);
            end = get_time();
            time1 = end - start;
            generate_line(i, "naive", time1);
            exit(EXIT_SUCCESS);
        }
        if (fork() == 0)
        {
            start = get_time();
            better_multiplication(A, B, C, i);
            end = get_time();
            time2 = end - start;
            generate_line(i, "better", time2);
            exit(EXIT_SUCCESS);
        }
        if (fork() == 0)
        {
            start = get_time();
            blas_multiplication(a, b, c, i);
            end = get_time();
            time3 = end - start;
            generate_line(i, "blas", time3);
            exit(EXIT_SUCCESS);
        }
        else
        {
            wait(NULL);
        }

        free_matrix(&A, &B, &C, &a, &b, &c, i);
    }
}

int main()
{
    test();
    return 0;
}
