# kNN classification

* Метрический классификатор kNN, k - количество проверяемых ближайших соседей
* Датасет chips представляет собой набор объектов, каждый из которых точка (x,y) и класс {0,1}
* Для настройки классификатора используется F1-мера
* Требуется >=2 пространственных преобразований, >=2 ядер и >=2 метрики для настройки kNN

## Highest F score achieved
Since the data set gets shuffled, the results are different each time

### Result 1
* k=3, batch_count=4
* Accuracy: 0.793103
* Precision: 0.774194
* Recall: 0.827586
* F-measure: 0.800000

### Result 2
* k=9, batch_count=40
* Accuracy: 0.775000
* Precision: 0.800000
* Recall: 0.761905
* F-measure: 0.780488
