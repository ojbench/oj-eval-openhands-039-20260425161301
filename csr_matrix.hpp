
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

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

// TODO: Implement a CSR matrix class
// You only need to implement the TODOs in this file
// DO NOT modify other parts of this file
// DO NOT include any additional headers
// DO NOT use STL other than std::vector

template <typename T>
class CSRMatrix {

private:
    // TODO: add your private member variables here
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
    // TODO: Initialize an empty CSR matrix with n rows and m columns
    CSRMatrix(size_t n, size_t m);

    // Constructor with pre-built CSR components
    // TODO: Initialize CSR matrix from existing CSR format data, validate sizes
    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr, 
        const std::vector<size_t> &indices,
        const std::vector<T> &data);

    // Copy constructor
    CSRMatrix(const CSRMatrix &other) = default;

    // Move constructor
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor from dense matrix format (given as vector of vectors)
    // TODO: Convert dense matrix representation to CSR format
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &data);

    // Destructor
    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    // TODO: Return the number of rows
    size_t getRowSize() const;

    // TODO: Return the number of columns
    size_t getColSize() const;

    // TODO: Return the count of non-zero elements
    size_t getNonZeroCount() const;

    // Element access
    // TODO: Retrieve element at position (i,j)
    T get(size_t i, size_t j) const;

    // TODO: Set element at position (i,j), updating CSR structure as needed
    void set(size_t i, size_t j, const T &value);

    // Access CSR components
    // TODO: Return the row pointer array
    const std::vector<size_t> &getIndptr() const;

    // TODO: Return the column indices array
    const std::vector<size_t> &getIndices() const;

    // TODO: Return the data values array
    const std::vector<T> &getData() const;

    // Convert to dense matrix format
    // TODO: Convert CSR format to dense matrix representation
    std::vector<std::vector<T>> getMatrix() const;

    // Matrix-vector multiplication
    // TODO: Implement multiplication of this matrix with vector vec
    std::vector<T> operator*(const std::vector<T> &vec) const;

    // Row slicing
    // TODO: Extract submatrix containing rows [l,r)
    CSRMatrix getRowSlice(size_t l, size_t r) const;
};

}

#endif // CSR_MATRIX_HPP
