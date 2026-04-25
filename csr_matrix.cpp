

#include "csr_matrix.hpp"

namespace sjtu {

// Constructor for empty matrix with dimensions
template <typename T>
CSRMatrix<T>::CSRMatrix(size_t n, size_t m) : rows(n), cols(m) {
    indptr.resize(n + 1, 0);
    // indices and data remain empty for zero matrix
}

// Constructor with pre-built CSR components
template <typename T>
CSRMatrix<T>::CSRMatrix(size_t n, size_t m, size_t count,
    const std::vector<size_t> &indptr, 
    const std::vector<size_t> &indices,
    const std::vector<T> &data) : rows(n), cols(m), indptr(indptr), indices(indices), data(data) {
    
    // Validate sizes
    if (indptr.size() != n + 1) {
        throw size_mismatch();
    }
    if (indices.size() != count || data.size() != count) {
        throw size_mismatch();
    }
    if (indptr[n] != count) {
        throw size_mismatch();
    }
}

// Constructor from dense matrix format
template <typename T>
CSRMatrix<T>::CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &dense_data) : rows(n), cols(m) {
    indptr.resize(n + 1, 0);
    
    // First pass: count non-zero elements per row
    size_t total_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (dense_data[i].size() != m) {
            throw size_mismatch();
        }
        for (size_t j = 0; j < m; j++) {
            if (dense_data[i][j] != T()) {
                total_count++;
            }
        }
        indptr[i + 1] = total_count;
    }
    
    // Second pass: fill indices and data
    indices.reserve(total_count);
    data.reserve(total_count);
    
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            if (dense_data[i][j] != T()) {
                indices.push_back(j);
                data.push_back(dense_data[i][j]);
            }
        }
    }
}

// Get dimensions and non-zero count
template <typename T>
size_t CSRMatrix<T>::getRowSize() const {
    return rows;
}

template <typename T>
size_t CSRMatrix<T>::getColSize() const {
    return cols;
}

template <typename T>
size_t CSRMatrix<T>::getNonZeroCount() const {
    return data.size();
}

// Element access
template <typename T>
T CSRMatrix<T>::get(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw invalid_index();
    }
    
    // Search for element (i,j) in row i
    size_t start = indptr[i];
    size_t end = indptr[i + 1];
    
    for (size_t k = start; k < end; k++) {
        if (indices[k] == j) {
            return data[k];
        }
    }
    
    // Element not found, return default constructed value
    return T();
}

template <typename T>
void CSRMatrix<T>::set(size_t i, size_t j, const T &value) {
    if (i >= rows || j >= cols) {
        throw invalid_index();
    }
    
    size_t start = indptr[i];
    size_t end = indptr[i + 1];
    
    // Search for existing element
    for (size_t k = start; k < end; k++) {
        if (indices[k] == j) {
            // Element found, update it
            data[k] = value;
            return;
        }
    }
    
    // Element not found, insert it
    // Find insertion position to maintain sorted order
    size_t insert_pos = start;
    while (insert_pos < end && indices[insert_pos] < j) {
        insert_pos++;
    }
    
    // Insert new element
    indices.insert(indices.begin() + insert_pos, j);
    data.insert(data.begin() + insert_pos, value);
    
    // Update indptr for all subsequent rows
    for (size_t r = i + 1; r <= rows; r++) {
        indptr[r]++;
    }
}

// Access CSR components
template <typename T>
const std::vector<size_t> &CSRMatrix<T>::getIndptr() const {
    return indptr;
}

template <typename T>
const std::vector<size_t> &CSRMatrix<T>::getIndices() const {
    return indices;
}

template <typename T>
const std::vector<T> &CSRMatrix<T>::getData() const {
    return data;
}

// Convert to dense matrix format
template <typename T>
std::vector<std::vector<T>> CSRMatrix<T>::getMatrix() const {
    std::vector<std::vector<T>> result(rows, std::vector<T>(cols, T()));
    
    for (size_t i = 0; i < rows; i++) {
        size_t start = indptr[i];
        size_t end = indptr[i + 1];
        
        for (size_t k = start; k < end; k++) {
            size_t j = indices[k];
            result[i][j] = data[k];
        }
    }
    
    return result;
}

// Matrix-vector multiplication
template <typename T>
std::vector<T> CSRMatrix<T>::operator*(const std::vector<T> &vec) const {
    if (vec.size() != cols) {
        throw size_mismatch();
    }
    
    std::vector<T> result(rows, T());
    
    for (size_t i = 0; i < rows; i++) {
        size_t start = indptr[i];
        size_t end = indptr[i + 1];
        
        for (size_t k = start; k < end; k++) {
            size_t j = indices[k];
            result[i] += data[k] * vec[j];
        }
    }
    
    return result;
}

// Row slicing
template <typename T>
CSRMatrix<T> CSRMatrix<T>::getRowSlice(size_t l, size_t r) const {
    if (l > r || r > rows) {
        throw invalid_index();
    }
    
    size_t new_rows = r - l;
    CSRMatrix<T> result(new_rows, cols);
    
    // Copy indptr (adjusting for slice)
    result.indptr.resize(new_rows + 1);
    size_t offset = indptr[l];
    for (size_t i = 0; i <= new_rows; i++) {
        result.indptr[i] = indptr[l + i] - offset;
    }
    
    // Copy indices and data
    size_t start_idx = indptr[l];
    size_t end_idx = indptr[r];
    result.indices.assign(indices.begin() + start_idx, indices.begin() + end_idx);
    result.data.assign(data.begin() + start_idx, data.begin() + end_idx);
    
    return result;
}

// Explicit template instantiations for common types
template class CSRMatrix<int>;
template class CSRMatrix<double>;
template class CSRMatrix<float>;

}
