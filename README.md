# Nuget Master
1) Place saved model file in a  directory _nuget_master/src/nuget-master/src/python-scripts/model/trained_models_
2) Set "auth" variable **personal access tokens (classic)**. Or you can remove it, but it may lead to GitHub rate limits after a while
3) Now, change working directory for ./src/nuget-master and run the project

https://github.com/razrez/nuget-master/assets/70781439/7c8ff54b-992e-4fce-8874-c382fde70ffd

# Плагин Visual Studio Code для рекомендации NuGet зависимостей на основе машинного обучения
____
Модель берёт текстовое описание (проекта/запроса), превращает его в вектор (с помощью BERT), а затем с помощью kNN находит наиболее близкие по смыслу репозитории из датасета. Идея в том, чтобы по текстовому описанию рекомендовать NuGet-зависимости, подходящие под задачи проекта. Сначала идёт сбор/очистка данных (убираем эмоджи, «мусор», приводим к единому формату), потом векторизация (BERT), а далее построение kNN для быстрых рекомендаций по схожести текстовых описаний.
