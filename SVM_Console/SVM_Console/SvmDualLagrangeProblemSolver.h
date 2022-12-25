#pragma once
#include "Data.h"
#include "MathFuncions.h"

using namespace math_functions;

class SvmDualLagrangeProblemSolver
{
public:
	/// <summary>
	/// ��������� ����������� �������� ���������� �������� ��� ������������ ������ ������ ���������� ����������� ��������������
	/// </summary>
	/// <param name="training_input">��������� �������</param>
	/// <param name="kernel">���� (�� ��������� - ��������� ������������)</param>
	/// <returns>������ ���������� ��������</returns>
	static std::vector<number_elem_t> get_optimal_lagrange_multiplifiers
	(
		const std::vector<TrainingDataVector>& training_input,
		const kernel_t& kernel = math_functions::scalar_product
	);
};
