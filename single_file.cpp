

#include <vector>
#include <exception>
#include <iostream>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override {
        return "Size mismatch";
    }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override {
        return "Index out of range";
    }
};

template <typename T>
class CSRMatrix {

private:
    size_t rows;
    size_t cols;
    std::vector<size_t> indptr;
    std::vector<size_t> indices;
    std::vector<T> data;
    
public:
    // Assignment operators are deleted
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    // Constructor for empty matrix with dimensions
    CSRMatrix(size_t n, size_t m);

    // Constructor with pre-built CSR components
    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr, 
        const std::vector<size_t> &indices,
        const std::vector<T> &data);

    // Copy constructor
    CSRMatrix(const CSRMatrix &other) = default;

    // Move constructor
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor from dense matrix format (given as vector of vectors)
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &data);

    // Destructor
    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    size_t getRowSize() const;
    size_t getColSize() const;
    size_t getNonZeroCount() const;

    // Element access
    T get(size_t i, size_t j) const;
    void set(size_t i, size_t j, const T &value);

    // Access CSR components
    const std::vector<size_t> &getIndptr() const;
    const std::vector<size_t> &getIndices() const;
    const std::vector<T> &getData() const;

    // Convert to dense matrix format
    std::vector<std::vector<T>> getMatrix() const;

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T> &vec) const;

    // Row slicing
    CSRMatrix getRowSlice(size_t l, size_t r) const;
};

// Implementation

template <typename T>
CSRMatrix<T>::CSRMatrix(size_t n, size_t m) : rows(n), cols(m) {
    indptr.resize(n + 1, 0);
}

template <typename T>
CSRMatrix<T>::CSRMatrix(size_t n, size_t m, size_t count,
    const std::vector<size_t> &indptr, 
    const std::vector<size_t> &indices,
    const std::vector<T> &data) : rows(n), cols(m), indptr(indptr), indices(indices), data(data) {
    
    // Validate indptr size
    if (indptr.size() != n + 1) {
        throw size_mismatch();
    }
    
    // Validate indices and data sizes
    if (indices.size() != count || data.size() != count) {
        throw size_mismatch();
    }
    
    // Validate that indptr[n] matches count
    if (indptr[n] != count) {
        throw size_mismatch();
    }
    
    // Validate that indptr is non-decreasing
    for (size_t i = 0; i < n; i++) {
        if (indptr[i] > indptr[i + 1]) {
            throw size_mismatch();
        }
    }
    
    // Validate that column indices are within bounds
    for (size_t i = 0; i < count; i++) {
        if (indices[i] >= m) {
            throw invalid_index();
        }
    }
}

template <typename T>
CSRMatrix<T>::CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &dense_data) : rows(n), cols(m) {
    if (dense_data.size() != n) {
        throw size_mismatch();
    }
    
    indptr.resize(n + 1, 0);
    
    // First pass: count non-zero elements and validate input
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
    
    // Reserve space to avoid reallocations
    indices.reserve(total_count);
    data.reserve(total_count);
    
    // Second pass: fill indices and data
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            if (dense_data[i][j] != T()) {
                indices.push_back(j);
                data.push_back(dense_data[i][j]);
            }
        }
    }
}

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

template <typename T>
T CSRMatrix<T>::get(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw invalid_index();
    }
    
    size_t start = indptr[i];
    size_t end = indptr[i + 1];
    
    // Binary search for better performance
    while (start < end) {
        size_t mid = start + (end - start) / 2;
        if (indices[mid] == j) {
            return data[mid];
        } else if (indices[mid] < j) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    
    return T();
}

template <typename T>
void CSRMatrix<T>::set(size_t i, size_t j, const T &value) {
    if (i >= rows || j >= cols) {
        throw invalid_index();
    }
    
    size_t start = indptr[i];
    size_t end = indptr[i + 1];
    
    // Binary search for existing element
    size_t left = start, right = end;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        if (indices[mid] == j) {
            data[mid] = value;
            return;
        } else if (indices[mid] < j) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    // Element not found, insert it at position left
    indices.insert(indices.begin() + left, j);
    data.insert(data.begin() + left, value);
    
    // Update indptr for all subsequent rows
    for (size_t r = i + 1; r <= rows; r++) {
        indptr[r]++;
    }
}

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

template <typename T>
CSRMatrix<T> CSRMatrix<T>::getRowSlice(size_t l, size_t r) const {
    if (l > r || r > rows) {
        throw invalid_index();
    }
    
    size_t new_rows = r - l;
    CSRMatrix<T> result(new_rows, cols);
    
    result.indptr.resize(new_rows + 1);
    size_t offset = indptr[l];
    for (size_t i = 0; i <= new_rows; i++) {
        result.indptr[i] = indptr[l + i] - offset;
    }
    
    size_t start_idx = indptr[l];
    size_t end_idx = indptr[r];
    result.indices.assign(indices.begin() + start_idx, indices.begin() + end_idx);
    result.data.assign(data.begin() + start_idx, data.begin() + end_idx);
    
    return result;
}

// Explicit template instantiations
template class CSRMatrix<int>;
template class CSRMatrix<double>;
template class CSRMatrix<float>;

}

