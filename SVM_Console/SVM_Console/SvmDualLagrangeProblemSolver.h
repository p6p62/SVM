#pragma once
#include "Data.h"
#include "MathFuncions.h"

using namespace math_functions;

class SvmDualLagrangeProblemSolver
{
public:
	/// <summary>
	/// Вычисляет оптимальные значения множителей Лагранжа для двойственной задачи поиска параметров оптимальной гиперплоскости
	/// </summary>
	/// <param name="training_input">Обучающая выборка</param>
	/// <param name="kernel">Ядро (по умолчанию - скалярное произведение)</param>
	/// <returns>Вектор множителей Лагранжа</returns>
	static std::vector<number_elem_t> get_optimal_lagrange_multiplifiers
	(
		const std::vector<TrainingDataVector>& training_input,
		const kernel_t& kernel = math_functions::scalar_product
	);
};
