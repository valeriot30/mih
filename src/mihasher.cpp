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

void mihasher::batchquery(UINT32 *results, UINT32 *numres, qstat *stats, UINT8 *queries, UINT32 numq, int dim1queries)
{
    counter = new bitarray;
    counter->init(N);

    UINT32 *res  = new UINT32[K*(D+1)];
    UINT64 *chunks = new UINT64[m];

    UINT32 *presults = results;
    UINT32 *pnumres = numres;
    qstat *pstats = stats;
    UINT8 *pq = queries;

    
    for (int i=0; i<numq; i++) {
	query(presults, pnumres, pstats, pq, chunks, res);

	presults += K;
	pnumres += B+1;
	pstats ++;
	pq += dim1queries;
    }
	
    delete [] res;
    delete [] chunks;

    delete counter;
}



// Temp variables: chunks, res -- I did not want to malloc inside
// query, so these arrays are passed from outside
void mihasher::query(UINT32 *results, UINT32 *numres, qstat *stats, UINT8 *query, UINT64 *chunks, UINT32 *res) {
    UINT32 maxres = K ? K : N; // if K == 0 that means we want everything to be processed.
    // So maxres = N in that case. Otherwise K limits the results processed.

    UINT32 n = 0; // number of results so far obtained (up to a distance of s per chunk)
    UINT32 nc = 0; // number of candidates tested with full codes (not counting duplicates)
    UINT32 nd = 0; // counting everything retrieved (duplicates are counted multiple times)
    UINT32 nl = 0; // number of lookups (and xors)
    clock_t start, end;

    start = clock();

    counter->erase();
    memset(numres, 0, (B + 1) * sizeof(*numres));

    // Splitta la query nei chunks
    split(chunks, query, m, mplus, b);

    for (int s = 0; s <= d && n < maxres; s++) {
        bool stop = false;

        #pragma omp parallel for shared(stop) schedule(dynamic) 
        for (int k = 0; k < m; k++) {
            if (stop) continue;

            int local_curb = (k < mplus) ? b : (b - 1);

            // Il k-esimo chunk di ricerca
            UINT64 chunksk = chunks[k];
            UINT64 bitstr = 0; // the bit-string with s number of 1s
            vector<int> local_power(s + 1);
            for (int i = 0; i < s; i++)
                local_power[i] = i; // power[i] stores the location of the i'th 1
            local_power[s] = local_curb + 1; // used for stopping criterion (location of (s+1)th 1)

            int bit = s - 1; // bit determines the 1 that should be moving to the left
            // We start from the left-most 1, and move it to the left until it touches another one

            while (true) { // the loop for changing bitstr
                if (bit != -1) {
                    bitstr ^= (local_power[bit] == bit) ? (UINT64)1 << local_power[bit] : (UINT64)3 << (local_power[bit] - 1);
                    local_power[bit]++;
                    bit--;
                } else { // bit == -1
                    int size;
                    UINT32* arr = H[k].query(chunksk ^ bitstr, &size); // lookup
                    if (size) { // the corresponding bucket is not empty
                        for (int c = 0; c < size; c++) {
                            int index = arr[c];
                            bool is_duplicate;
                                is_duplicate = counter->get(index);
                                if (!is_duplicate) {
                                    counter->set(index);
                                }
                            if (!is_duplicate) {
                                int hammd = match(codes + (UINT64)index * (B_over_8), query, B_over_8);
                                if (hammd <= D && numres[hammd] < maxres) {
                                   res[hammd * K + numres[hammd]] = index + 1;
                                   nc++;
                                   numres[hammd]++;
                                }
                            }
                        }
                    }

                    while (++bit < s && local_power[bit] == local_power[bit + 1] - 1) {
                        bitstr ^= (UINT64)1 << (local_power[bit] - 1);
                        local_power[bit] = bit;
                    }
                    if (bit == s)
                        break;
                }
            }

            n += numres[s * m + k]; // This line is very tricky ;)
                if (n >= maxres) {
                    stop = true;
                }
        }
        if (stop) break;
    }

    end = clock();

    stats->ticks = end - start;
    stats->numcand = nc;
    stats->numdups = nd;
    stats->numlookups = nl;

    n = 0;

    for (int s = 0; s <= D; s++) {
        for (int c = 0; c < numres[s] && n < K; c++)
            results[n++] = res[s * K + c];
    }

    UINT32 total = 0;
    stats->maxrho = -1;
    #pragma omp for
    for (int i = 0; i <= B; i++) {
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
 
    D = ceil(B/2.0);		// assuming that B/2 is large enough radius to include all of the k nearest neighbors
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

	printf("Executing with OpenMP with %d threads ", omp_get_max_threads());
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
	    // 	H[k].allocate_mem_based_on_counts();
	    
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