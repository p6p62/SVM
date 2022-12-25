#pragma once
#include <tuple>
#include "Data.h"
#include "MathFuncions.h"

using namespace math_functions;

class BinaryLinearSVM
{
public:
	static constexpr double ZERO_ACCURACY{ 1e-8 };

public:
	struct ClassifierModel
	{
		// ����������� ������������ ���������
		size_t dimensionality{ 0 };

		// ������� ������� � �������� ���������� �������� ��� ���
		std::vector<std::pair<TrainingDataVector, number_elem_t>> support_vectors;
		number_elem_t offset{ 0 };
		kernel_t kernel{ scalar_product };
	};

private:
	ClassifierModel model_;

public:
	BinaryLinearSVM(const ClassifierModel& model);
	BinaryLinearSVM(const std::vector<TrainingDataVector>& training_samples, const kernel_t& kernel = scalar_product);

public:
	static constexpr bool check_training_samples(const std::vector<TrainingDataVector>& training_samples);
	void train_svm(const std::vector<TrainingDataVector>& training_samples, const kernel_t& kernel = scalar_product);
	ClassLabel classify(const DataVector& x) const;
};
