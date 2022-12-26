# SVM
### Алгоритм бинарной классификации на основе метода опорных векторов для НИР  
Работает с линейно-разделимыми данными (есть задел на использование функций ядра, но пока из них реализовано только скалярное произведение).
Обучение происходит по принципу жёсткого зазора

В проекте имеется приложение с консольным интерфейсом, позволяющее на базовом уровне обучать и использовать классификатор с данными в формете CSV
(параметры модели сохраняются в бинарном формате для переиспользования)

*Зависимости*:<br>
CGAL QP (https://doc.cgal.org/latest/QP_solver/index.html)