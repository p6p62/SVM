#include "MathFuncions.h"
#include <stdexcept>
#include <numeric>

DataVector& math_functions::multiplicate_vector(DataVector& x, number_elem_t multiplifier) noexcept
{
	for (size_t i{ 0 }; i < x.size(); ++i)
	{
		x[i] *= multiplifier;
	}
	return x;
}

DataVector& math_functions::sum_vectors(DataVector& x_dest, const DataVector& x1, const DataVector& x2)
{
	if (x1.size() == x2.size() && x1.size() == x_dest.size())
	{
		for (size_t i{ 0 }; i < x_dest.size(); ++i)
		{
			x_dest[i] = x1[i] + x2[i];
		}
	}
	else
	{
		throw std::invalid_argument("Ошибка, вектора разной длины");
	}
	return x_dest;
}

number_elem_t math_functions::scalar_product(const DataVector& x1, const DataVector& x2)
{
	return std::inner_product(x1.begin(), x1.end(), x2.begin(), 0.0);
}
