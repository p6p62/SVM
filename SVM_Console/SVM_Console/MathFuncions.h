#pragma once
#include "Data.h"

namespace math_functions
{
	DataVector& multiplicate_vector(DataVector& x, number_elem_t multiplifier) noexcept;
	DataVector& sum_vectors(DataVector& x_dest, const DataVector& x1, const DataVector& x2);
	number_elem_t scalar_product(const DataVector& x1, const DataVector& x2);
}
