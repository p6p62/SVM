#include <iostream>
#include "SvmDualLagrangeProblemSolver.h"

constexpr char CLASS_FIRST = 1;
constexpr char CLASS_SECOND = -1;

void get_data(std::vector<TrainingDataVector>& out_data)
{
	out_data.push_back({ {1, 1}, CLASS_FIRST });
	out_data.push_back({ {1, -1}, CLASS_FIRST });
	out_data.push_back({ {-1, 1}, CLASS_SECOND });
	out_data.push_back({ {-1, -1}, CLASS_SECOND });
}

int main()
{
	std::cout << "Hello World!\n";
	std::vector<TrainingDataVector> data;
	get_data(data);
	SvmDualLagrangeProblemSolver::get_optimal_lagrange_multiplifiers(data);
}
