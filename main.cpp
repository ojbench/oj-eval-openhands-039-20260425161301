


#include "csr_matrix.hpp"
#include <iostream>

using namespace sjtu;

int main() {
    try {
        // Test basic functionality
        std::cout << "Testing CSR Matrix implementation..." << std::endl;
        
        // Test 1: Empty matrix constructor
        CSRMatrix<int> mat1(3, 4);
        std::cout << "Empty matrix: " << mat1.getRowSize() << "x" << mat1.getColSize() 
                  << ", non-zero count: " << mat1.getNonZeroCount() << std::endl;
        
        // Test 2: Set and get operations
        mat1.set(0, 1, 5);
        mat1.set(1, 2, 10);
        mat1.set(2, 3, 15);
        
        std::cout << "After setting elements:" << std::endl;
        std::cout << "mat1(0,1) = " << mat1.get(0, 1) << std::endl;
        std::cout << "mat1(1,2) = " << mat1.get(1, 2) << std::endl;
        std::cout << "mat1(2,3) = " << mat1.get(2, 3) << std::endl;
        std::cout << "mat1(0,0) = " << mat1.get(0, 0) << " (should be 0)" << std::endl;
        std::cout << "Non-zero count: " << mat1.getNonZeroCount() << std::endl;
        
        // Test 3: Dense matrix constructor
        std::vector<std::vector<int>> dense = {
            {1, 0, 3},
            {0, 2, 0},
            {4, 0, 5}
        };
        CSRMatrix<int> mat2(3, 3, dense);
        std::cout << "\nDense matrix constructor:" << std::endl;
        std::cout << "Non-zero count: " << mat2.getNonZeroCount() << std::endl;
        
        // Test 4: Convert to dense
        auto dense_result = mat2.getMatrix();
        std::cout << "Converted back to dense:" << std::endl;
        for (const auto& row : dense_result) {
            for (int val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        
        // Test 5: Matrix-vector multiplication
        std::vector<int> vec = {1, 2, 3};
        auto result = mat2 * vec;
        std::cout << "\nMatrix-vector multiplication:" << std::endl;
        for (int val : result) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        // Test 6: Row slicing
        auto slice = mat2.getRowSlice(1, 3);
        std::cout << "\nRow slice (rows 1-2):" << std::endl;
        std::cout << "Slice dimensions: " << slice.getRowSize() << "x" << slice.getColSize() << std::endl;
        std::cout << "Slice non-zero count: " << slice.getNonZeroCount() << std::endl;
        
        std::cout << "\nAll tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

