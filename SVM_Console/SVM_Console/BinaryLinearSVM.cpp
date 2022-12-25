#include "BinaryLinearSVM.h"
#include "SvmDualLagrangeProblemSolver.h"
#include <stdexcept>
#include <algorithm>

using ClassLabel::FIRST;

BinaryLinearSVM::BinaryLinearSVM(const std::vector<TrainingDataVector>& training_samples, const kernel_t& kernel)
{
	train_svm(training_samples, kernel);
}

BinaryLinearSVM::BinaryLinearSVM(const ClassifierModel& model)
{
	if (model.dimensionality == 0 || model.support_vectors.empty())
	{
		throw std::runtime_error{ "Параметры модели заданы неверно!" };
	}
	this->model_ = model;
}

void BinaryLinearSVM::train_svm(const std::vector<TrainingDataVector>& training_samples, const kernel_t& kernel)
{
	if (!check_training_samples(training_samples))
		throw std::invalid_argument{ "Обучающая выборка пуста или содержит данные разной размерности!" };

	std::vector<number_elem_t> lagrange_multiplifiers{ SvmDualLagrangeProblemSolver::get_optimal_lagrange_multiplifiers(training_samples, kernel) };
	std::vector<size_t> support_vector_indexes;

	model_.dimensionality = training_samples[0].data_vector.size();
	model_.kernel = kernel;
	for (size_t i{ 0 }; i < lagrange_multiplifiers.size(); ++i)
		if (lagrange_multiplifiers[i] > ZERO_ACCURACY)
			model_.support_vectors.push_back({ training_samples[i], lagrange_multiplifiers[i] });

	// нахождение смещения
	const TrainingDataVector& x{ model_.support_vectors[0].first.data_vector };
	number_elem_t offset{ static_cast<double>(x.class_label) };
	const auto& sup_vecs{ model_.support_vectors };
	for (size_t i{ 0 }; i < sup_vecs.size(); ++i)
	{
		const TrainingDataVector& x_sup{ sup_vecs[i].first };
		offset -= kernel(x.data_vector, x_sup.data_vector) * static_cast<double>(x_sup.class_label) * sup_vecs[i].second;
	}
	model_.offset = offset;
}

constexpr bool BinaryLinearSVM::check_training_samples(const std::vector<TrainingDataVector>& training_samples)
{
	bool result{ training_samples.empty() };
	if (!result)
	{
		const size_t vector_size{ training_samples[0].data_vector.size() };
		result = std::all_of(training_samples.begin(), training_samples.end(), [vector_size](const TrainingDataVector& row) -> bool {return row.data_vector.size() == vector_size; });
	}
	return result;
}

ClassLabel BinaryLinearSVM::classify(const DataVector& x) const
{
	if (x.size() != model_.dimensionality)
		throw std::invalid_argument("Классификатор обучен на данных другой размерности!");

	double signed_distance{ model_.offset };
	for (const std::pair<TrainingDataVector, number_elem_t>& v_sup : model_.support_vectors)
	{
		const TrainingDataVector& x_sup{ v_sup.first };
		signed_distance += static_cast<double>(x_sup.class_label) * v_sup.second * model_.kernel(x, x_sup.data_vector);
	}
	return signed_distance >= 0 ? ClassLabel::FIRST : ClassLabel::SECOND;
}
