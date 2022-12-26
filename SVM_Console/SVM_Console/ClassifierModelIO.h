#pragma once
#include "BinaryLinearSVM.h"

/// <summary>
/// Формат хранения параметров классификатора:
///
/// [uint64_t]				dim - размерность данных
/// [char]					kernel - тип ядра
/// [double]				b - смещение
/// [uint64_t]				n - количество опорных векторов
/// [double] * dim			v1_data - данные опорного вектора 1
/// [char]					v1_class - метка класса опорного вектора 1
/// [double]				alpha_i - множитель Лагранжа для опорного вектора 1
/// ...
/// ...
/// ...
/// [double] * dim			v1_data - данные опорного вектора n
/// [char]					v1_class - метка класса опорного вектора n
/// [double]				alpha_i - множитель Лагранжа для опорного вектора n
/// </summary>
class ClassifierModelIO
{
public:
	static void save_svm_config(const BinaryLinearSVM::ClassifierModel& model, const std::string& filename);
	static BinaryLinearSVM::ClassifierModel load_svm_config(const std::string& filename);
};

