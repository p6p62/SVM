#pragma once
#include "BinaryLinearSVM.h"

/// <summary>
/// ������ �������� ���������� ��������������:
///
/// [uint64_t]				dim - ����������� ������
/// [char]					kernel - ��� ����
/// [double]				b - ��������
/// [uint64_t]				n - ���������� ������� ��������
/// [double] * dim			v1_data - ������ �������� ������� 1
/// [char]					v1_class - ����� ������ �������� ������� 1
/// [double]				alpha_i - ��������� �������� ��� �������� ������� 1
/// ...
/// ...
/// ...
/// [double] * dim			v1_data - ������ �������� ������� n
/// [char]					v1_class - ����� ������ �������� ������� n
/// [double]				alpha_i - ��������� �������� ��� �������� ������� n
/// </summary>
class ClassifierModelIO
{
public:
	static void save_svm_config(const BinaryLinearSVM::ClassifierModel& model, const std::string& filename);
	static BinaryLinearSVM::ClassifierModel load_svm_config(const std::string& filename);
};

