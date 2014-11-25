#ifndef DELL_MATRIX_H
#define DELL_MATRIX_H

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
struct dell_matrix         //dynamic ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> rs;             //row sizes
    cusp::array1d<INDEX_TYPE, MEM_TYPE> rm;             //row max
    cusp::array1d<INDEX_TYPE, MEM_TYPE> ci;             //index of next chunk
    cusp::array1d<INDEX_TYPE, MEM_TYPE> cl;             //chunk length
    cusp::array1d<INDEX_TYPE, MEM_TYPE> ca;             //chunk address
    cusp::array1d<INDEX_TYPE, MEM_TYPE> cols;           //columns
    cusp::array1d<INDEX_TYPE, MEM_TYPE> vals;           //values
    cusp::array1d<INDEX_TYPE, MEM_TYPE> Matrix_MD;

    float alpha;                //alpha threshold
    size_t pitch;
    size_t chunks;
    size_t chunk_size;          //chunk size
    size_t chunk_length;        //chunk length
    size_t mem_size;            //total memory used
    size_t num_rows;
    size_t num_cols;
    size_t num_entries;

    void resize(const size_t n_rows, const size_t n_cols, const size_t c_size, const size_t c_length)
    {
        size_t max_layers = 8;

        chunk_size = c_size;
        chunk_length = c_length;
        chunks = ceil(double(n_rows) / double(chunk_size));
        pitch = chunk_size;
        mem_size = pitch * c_length * chunks * max_layers;
        num_rows = n_rows;
        num_cols = n_cols;
        num_entries = 0;

        Matrix_MD.resize(4);
        ci.resize(chunks*max_layers);
        cl.resize(chunks*max_layers);
        ca.resize(chunks*max_layers);
        rs.resize(chunks*max_layers);
        rm.resize(chunks*max_layers, c_length);
        cols.resize(mem_size);
    }
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
struct dell_matrix_B         //dynamic ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> rs;             //row sizes
    cusp::array1d<INDEX_TYPE, MEM_TYPE> ci;             //index of next chunk
    cusp::array1d<INDEX_TYPE, MEM_TYPE> cl;             //chunk length
    cusp::array1d<INDEX_TYPE, MEM_TYPE> ca;             //chunk address
    cusp::array1d<INDEX_TYPE, MEM_TYPE> cols;           //columns
    cusp::array1d<INDEX_TYPE, MEM_TYPE> Matrix_MD;

    float alpha;                //alpha threshold
    size_t pitch;
    size_t init_chunks;         //number of initial chunks
    size_t chunk_size;          //chunk size
    size_t chunk_length;        //chunk length
    size_t mem_size;            //total memory used
    size_t num_rows;
    size_t num_cols;
    size_t num_entries;

    void resize(const size_t n_rows, const size_t n_cols, const size_t c_size, const size_t c_length)
    {
        size_t max_layers = 8;

        chunk_size = c_size;
        chunk_length = c_length;
        init_chunks = ceil(double(n_rows) / double(chunk_size));
        pitch = chunk_size;
        mem_size = pitch * c_length * init_chunks * max_layers;
        num_rows = n_rows;
        num_cols = n_cols;
        num_entries = 0;

        fprintf(stderr, "rows: %d  cols: %d\n", n_rows, n_cols);
        fprintf(stderr, "init_chunks: %d\n", init_chunks);
        fprintf(stderr, "chunk_size: %d\n", chunk_size);
        fprintf(stderr, "chunk_length: %d\n", chunk_length);
        fprintf(stderr, "mem_size: %d\n", mem_size);

        Matrix_MD.resize(4);
        ci.resize(init_chunks*max_layers);
        cl.resize(init_chunks*max_layers);
        ca.resize(init_chunks*max_layers);
        rs.resize(num_rows);
        cols.resize(mem_size);
    }
};

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE>
struct hyb_matrix         //hyb ELL matrix
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> rs;             //row sizes
    cusp::hyb_matrix<INDEX_TYPE, VALUE_TYPE, MEM_TYPE> matrix;

    size_t num_rows;
    size_t num_cols;
    size_t num_entries;

    void resize(const size_t rows, const size_t cols, const size_t num_ell_entries, const size_t num_cols_per_row, const size_t num_coo_entries = DEFAULT_OVERFLOW)
    {
        num_rows = rows;
        num_cols = cols;
        num_entries = num_ell_entries;

        matrix.resize(rows, cols, num_ell_entries, num_coo_entries, num_cols_per_row);
        rs.resize(rows);
    }
};

#include "primitives.h"

#endif