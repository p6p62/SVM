#include "SvmDualLagrangeProblemSolver.h"
#include <numeric>
#include <algorithm>

using matrix_t = SvmDualLagrangeProblemSolver::matrix_t;

number_elem_t scalar_product(const DataVector& x1, const DataVector& x2)
{
	return std::inner_product(x1.begin(), x1.end(), x2.begin(), 0);
}

number_elem_t lagrangian_for_dual(const std::vector<number_elem_t>& lagrange_multiplifiers, const matrix_t& hessi_matrix)
{
	const std::vector<number_elem_t>& a{ lagrange_multiplifiers };
	number_elem_t result{ 0 };
	for (auto i{ 0 }; i < a.size(); ++i)
		for (auto j{ 0 }; j < a.size(); ++j)
			result -= a[i] * a[j] * hessi_matrix[i][j];
	result /= 2;

	for (auto i{ 0 }; i < a.size(); ++i)
		result += a[i];
	return result;
}

void calculate_hessi_matrix(const std::vector<TrainingDataVector>& training_input, std::vector<std::vector<number_elem_t>>& out_hessi_matrix)
{
	const auto& X{ training_input };
	for (auto i{ 0 }; i < X.size(); ++i)
	{
		for (auto j{ 0 }; j < X.size(); ++j)
		{
			out_hessi_matrix[i][j] = X[i].class_label * X[j].class_label
				* scalar_product(X[i].data_vector, X[j].data_vector);
		}
	}
}

bool is_zero(const DataVector& vector)
{
	return std::all_of(vector.begin(), vector.end(),
		[](number_elem_t x_i) -> bool
		{return abs(x_i) < SvmDualLagrangeProblemSolver::ZERO_ACCURACY; });
}

std::vector<number_elem_t> gradient(const DataVector& x)
{
	return { 0 };
}

std::vector<number_elem_t> SvmDualLagrangeProblemSolver::get_optimal_lagrange_multiplifiers(const std::vector<TrainingDataVector>& training_input)
{
	std::vector<number_elem_t> lagrange_multiplifiers(training_input.size(), 0);

	matrix_t hessi_matrix;
	calculate_hessi_matrix(training_input, hessi_matrix);

	while (!is_zero(gradient(lagrange_multiplifiers)))
	{

	}

	return lagrange_multiplifiers;
}
