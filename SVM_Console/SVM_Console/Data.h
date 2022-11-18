#pragma once
#include <vector>
#include <string>

using number_elem_t = double;
using DataVector = std::vector<number_elem_t>;

struct DataVectorsDescription
{
	std::vector<std::string> component_names;
};

struct TrainingDataVector
{
	static constexpr char UNDEFINED{ -1 };
	DataVector data_vector;
	char class_label{ UNDEFINED };
};
