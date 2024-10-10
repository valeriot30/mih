#include <algorithm>
#include "mihasher.h"

using namespace std;

/*
 * Inputs: query, numq, dim1queries
 */

/*
 * Outputs: results, numres, stats
 *
 *   results: an array of indices (1-based) of the K-nearest neighbors
 *   for each query. So the array includes K*numq uint32 integers.
 *
 *   numres: includes the number of database entries that fall at any
 *   specific Hamming distance from the query until the K nearest
 *   neighbors are reached. So from this array you can figure out the
 *   Hamming distances of the K-nearest neighbors.
 */

void mihasher::batchquery(UINT32 *results, UINT32 *numres, qstat *stats, UINT8 * q, UINT32 numq, int dim1queries) {

    printf("Batching %d queries...\n", numq);

    double wt2, wt3;
    #pragma omp parallel
    {
        #pragma omp single
        {
            double wt0 = omp_get_wtime();
            counter = new bitarray;
            counter->init(N);
            double wt1 = omp_get_wtime();

            printf("serial time for batchquery is: %f\n", wt1 - wt0);
        }
    }
    
    wt2 = omp_get_wtime();
    #pragma omp parallel
    {
        // Risorse locali per ogni thread
        UINT32 *res  = new UINT32[K*(D+1)];
        UINT64 *chunks = new UINT64[m];

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < numq; i++) {
            UINT32 *presults = results + i * K;
            UINT32 *pnumres = numres + i * (B+1);
            qstat *pstats = stats + i;
            UINT8 *pq = q + i * dim1queries;

            query(presults, pnumres, pstats, pq, chunks, res);
        }

        // Pulizia delle risorse locali
        delete [] res;

        delete [] chunks;
    }

    // Pulizia delle risorse condivise
    #pragma omp parallel
    {
        #pragma omp single
        {
            delete counter;
        }
    }
    wt3 = omp_get_wtime();

    printf("parallel time for batching queries is %f\n", wt3 - wt2);
}


// Temp variables: chunks, res -- I did not want to malloc inside
// query, so these arrays are passed from outside

void mihasher::query(UINT32 *results, UINT32* numres, qstat *stats, UINT8 *q, UINT64 * chunks, UINT32 * res) {
    UINT32 maxres = K ? K : N;          // if K == 0 that means we want everything to be processed.
                        // So maxres = N in that case. Otherwise K limits the results processed.

    UINT32 n = 0;               // number of results so far obtained (up to a distance of s per chunk)
    UINT32 nc = 0;              // number of candidates tested with full codes (not counting duplicates)
    UINT32 nd = 0;              // counting everything retrieved (duplicates are counted multiple times)
    UINT32 nl = 0;              // number of lookups (and xors)
    UINT32 *arr;
    int size = 0;
    UINT32 index;
    int hammd;
    clock_t start, end;

    start = clock();
    counter->erase();
    memset(numres, 0, (B+1)*sizeof(*numres));

    split(chunks, q, m, mplus, b);
    
    int s;          // the growing search radius per substring

    int curb = b;       // current b: for the first mplus substrings it is b, for the rest it is (b-1)

   double startwt0 = omp_get_wtime();
bool early_exit = false;
#pragma omp parallel shared(early_exit)
{
    #pragma omp single
    {
        for (s = 0; s <= d && !early_exit; s++) {            int local_n = 0;
            #pragma omp taskgroup
            {
                for (int k = 0; k < m; k++) {
                    #pragma omp task shared(local_n, early_exit)
                    {
                        int curb = (k < mplus) ? b : b-1;
                        UINT64 chunksk = chunks[k];
                        
                        nl += xornum[s+1] - xornum[s];

                        UINT64 bitstr = 0;
                        int power[curb+1];
                        for (int i = 0; i < s; i++)
                            power[i] = i;
                        power[s] = curb+1;
                        int bit = s-1;

                        while (true && !early_exit) {
                            if (bit != -1) {
                                bitstr ^= (power[bit] == bit) ? (UINT64)1 << power[bit] : (UINT64)3 << (power[bit]-1);
                                power[bit]++;
                                bit--;
                            } else {
                                int size;
                                UINT32* arr = H[k].query(chunksk ^ bitstr, &size);
                                if (size) {
                                    nd += size;

                                    for (int c = 0; c < size && !early_exit; c++) {
                                        UINT32 index = arr[c];
                                        bool not_duplicate;
                                        
                                        #pragma omp critical
                                        {
                                            not_duplicate = !counter->get(index);
                                            if (not_duplicate) {
                                                counter->set(index);
                                            }
                                        }

                                        if (not_duplicate) {
                                            int hammd = match(codes + (UINT64)index*(B_over_8), q, B_over_8);
                                            

                                            if (hammd <= D) {
                                                #pragma omp critical
                                                {
                                                    if (numres[hammd] < maxres) {
                                                        res[hammd * K + numres[hammd]] = index + 1;
                                                        numres[hammd]++;
                                                        local_n++;
                                                        
                                                        if (n + local_n >= maxres) {
                                                            early_exit = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                while (++bit < s && power[bit] == power[bit+1]-1) {
                                    bitstr ^= (UINT64)1 << (power[bit]-1);
                                    power[bit] = bit;
                                }
                                if (bit == s)
                                    break;
                            }
                        }
                    } // end of task
                } // end of for k
            } // end of taskgroup

            n += local_n;

            if (early_exit) break;
        } // end of for s
    } // end of single
} // end of parallel
    double endwt0 = omp_get_wtime();
    end = clock();

    printf("time for a query: %f\n", endwt0 - startwt0);

    stats->ticks = end-start;
    stats->numcand = nc;
    stats->numdups = nd;
    stats->numlookups = nl;

    n = 0;
    for (s = 0; s <= D && n < K; s++ ) {
        for (int c = 0; c < numres[s] && n < K; c++) {
            results[n] = res[s*K + c];
            //distances[n++] = dist[s*K +c];
            //printf("#%d#", dist[s*K +c]);
            //fflush(stdout);
        }
    }

    UINT32 total = 0;
    stats->maxrho = -1;
    for (int i=0; i<=B; i++) {
        total += numres[i];
    if (total >= K && stats->maxrho == -1)
        stats->maxrho = i;
    }
    stats->numres = n;
}

mihasher::mihasher(int _B, int _m)
{
    B = _B;
    B_over_8 = B/8;
    m = _m;
    b = ceil((double)B/m);
 
    // SALVO SET D=4 bit
    D = ceil(B/2.0);        // assuming that B/2 is large enough radius to include all of the k nearest neighbors
    
    d = ceil((double)D/m);
   
    mplus = B - m * (b-1);
    // mplus     is the number of chunks with b bits
    // (m-mplus) is the number of chunks with (b-1) bits

    xornum = new UINT32 [d+2];
    xornum[0] = 0;
    for (int i=0; i<=d; i++)
    xornum[i+1] = xornum[i] + choose(b, i);
    
    H = new SparseHashtable[m];
    // H[i].init might fail
    for (int i=0; i<mplus; i++)
    H[i].init(b);
    for (int i=mplus; i<m; i++)
    H[i].init(b-1);
}

void mihasher::setK(int _K)
{
    K = _K;
}

mihasher::~mihasher()
{
    delete[] xornum;
    delete[] H;
}

void mihasher::populate(UINT8 *_codes, UINT32 _N, int dim1codes)
{
    N = _N;
    codes = _codes;

    #pragma omp parallel
    {

        UINT64 * chunks = new UINT64[m];

        double wtime = omp_get_wtime();
        #pragma omp for
        for (int k=0; k<m; k++) 
        {

            UINT8 * pcodes = codes;

            for (UINT64 i=0; i<N; i++) 
            {
                split(chunks, pcodes, m, mplus, b);
                
                H[k].count_insert(chunks[k], i);

                if (i % (int)ceil((double)N/1000) == 0) {
                    printf("%.2f%%\r", (double)i/N * 100);
                    fflush(stdout);
                }
                pcodes += dim1codes;
            }

        // for (int k=0; k<m; k++)
        //  H[k].allocate_mem_based_on_counts();
        
            pcodes = codes;

            for (UINT64 i=0; i<N; i++) 
            {
                split(chunks, pcodes, m, mplus, b);
            
            //#pragma omp parallel 
            //{
                H[k].data_insert(chunks[k], i);

                if (i % (int)ceil((double)N/1000) == 0) {
                    printf("%.2f%%\r", (double)i/N * 100);
                    fflush(stdout);
                }
                pcodes += dim1codes;
            }
        //}
        }

    wtime = omp_get_wtime() - wtime;
    printf( "Time taken by thread %d is %f\n", omp_get_thread_num(), wtime );
     
    delete [] chunks;
}
}
