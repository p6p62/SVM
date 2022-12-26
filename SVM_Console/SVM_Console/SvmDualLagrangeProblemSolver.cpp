#include "SvmDualLagrangeProblemSolver.h"

#include <CGAL/QP_functions.h>
#include <CGAL/QP_models.h>
#include <CGAL/MP_Float.h>

#include <algorithm>
#include <iostream>

using namespace math_functions;
using matrix_num = matrix_t<number_elem_t>;

// подсчёт матрицы Гессе для целевой функции задачи поиска (её вторые производные по множителям Лагранжа)
void calculate_hessi_matrix
(
	const std::vector<TrainingDataVector>& training_input,
	matrix_num& out_hessi_matrix,
	const kernel_t& kernel
)
{
	const auto& X{ training_input };
	for (size_t i{ 0 }; i < X.size(); ++i)
	{
		for (size_t j{ 0 }; j < X.size(); ++j)
		{
			out_hessi_matrix[i][j] = static_cast<double>(X[i].class_label) * static_cast<double>(X[j].class_label)
				* kernel(X[i].data_vector, X[j].data_vector);
		}
	}
}

std::vector<number_elem_t> SvmDualLagrangeProblemSolver::get_optimal_lagrange_multiplifiers
(
	const std::vector<TrainingDataVector>& training_input,
	const kernel_t& kernel
)
{
	const size_t vectors_count{ training_input.size() };

	matrix_num hessi_matrix;
	hessi_matrix.assign(vectors_count, DataVector(vectors_count));
	calculate_hessi_matrix(training_input, hessi_matrix, kernel);

	CGAL::Quadratic_program<number_elem_t> dual_lagrange_task{ CGAL::Sign::LARGER };

	// целевая функция
	// alpha - множители Лагранжа
	// l - размер обучающей выборки
	// SUM(i=1,l; alpha_i) - 1/2 * SUM(i=1,l; SUM(j=1,l; alpha_i * a_j * hessi_i_j)) -> max по alpha
	for (size_t i{ 0 }; i < vectors_count; ++i)
	{
		dual_lagrange_task.set_c(i, -1);

		for (size_t j{ 0 }; j < vectors_count; ++j)
		{
			dual_lagrange_task.set_d(std::max(i, j), std::min(i, j), hessi_matrix[std::max(i, j)][std::min(i, j)]);
		}
	}

	// ограничение
	// y - метка класса
	// SUM(i=1,l; alpha_i * y_i) = 0
	for (size_t i{ 0 }; i < vectors_count; ++i)
	{
		dual_lagrange_task.set_a(i, 0, static_cast<double>(training_input[i].class_label));
	}
	dual_lagrange_task.set_b(0, 0);
	dual_lagrange_task.set_r(0, CGAL::Comparison_result::EQUAL);

	// ограничение на неотрицательность всех alpha
	for (size_t i{ 0 }; i < vectors_count; ++i)
	{
		dual_lagrange_task.set_l(i, true);
	}

	CGAL::Quadratic_program_solution<CGAL::MP_Float> solution;
	try
	{
		solution = CGAL::solve_quadratic_program(dual_lagrange_task, CGAL::MP_Float{});
	}
	catch (const std::exception&)
	{
		throw std::invalid_argument("\nОшибка при обучении классификатора! Возможно, выборка не является линейно-разделимой");
	}
	std::vector<number_elem_t> lagrange_multiplifiers;
	std::transform
	(
		solution.variable_values_begin(),
		solution.variable_values_end(),
		std::back_inserter(lagrange_multiplifiers),
		[](const auto& v) -> number_elem_t { return CGAL::to_double(v); }
	);
	return lagrange_multiplifiers;
}
