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
	enum class KernelType : char
	{
		SCALAR_PRODUCT
	};

	struct ClassifierModel
	{
		// размерность пространства признаков
		size_t dimensionality{ 0 };

		// опорные векторы и значени€ множителей Ћагранжа дл€ них
		std::vector<std::pair<TrainingDataVector, number_elem_t>> support_vectors;
		number_elem_t offset{ 0 };
		KernelType kernel_type{ KernelType::SCALAR_PRODUCT };
		kernel_t kernel{ scalar_product };
	};

private:
	ClassifierModel model_;

public:
	BinaryLinearSVM(const ClassifierModel& model);
	BinaryLinearSVM(const std::vector<TrainingDataVector>& training_samples, const kernel_t& kernel = scalar_product);

public:
	static constexpr void check_training_samples(const std::vector<TrainingDataVector>& training_samples);
	void train_svm(const std::vector<TrainingDataVector>& training_samples, const kernel_t& kernel = scalar_product);
	ClassLabel classify(const DataVector& x) const;
	ClassifierModel get_model() const noexcept { return model_; }
};
