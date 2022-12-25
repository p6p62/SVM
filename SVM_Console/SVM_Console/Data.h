#pragma once
#include <vector>
#include <string>
#include <functional>

template<typename T>
using matrix_t = std::vector<std::vector<T>>;

using number_elem_t = double;
using DataVector = std::vector<number_elem_t>;
using kernel_t = std::function<number_elem_t(const DataVector& x1, const DataVector& x2)>;

enum class ClassLabel : char
{
	FIRST = 1,
	SECOND = -1
};

struct DataVectorsDescription
{
	std::vector<std::string> component_names;
};

struct TrainingDataVector
{
	DataVector data_vector;
	ClassLabel class_label{ ClassLabel::FIRST };
};
