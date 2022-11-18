#pragma once
#include "Data.h"

class SvmDualLagrangeProblemSolver
{
public:
	using matrix_t = std::vector<std::vector<number_elem_t>>;

public:
	// ����������� ������� �������� ��� ���������� ���������� ������ 
	static constexpr number_elem_t ZERO_ACCURACY{ 1e-6 };

private:
	// ��� ���������� ���������� ��� ���������� ����������� � ����
	static constexpr number_elem_t VARIABLE_CHANGING_STEP{ ZERO_ACCURACY / 2 };

public:
	static std::vector<number_elem_t> get_optimal_lagrange_multiplifiers(const std::vector<TrainingDataVector>& training_input);
};
