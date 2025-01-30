# Nuget Master
1) Firstly, save the FastText model like this [here](https://git.kpfu.ru/RIKhodzhamirzoev/nuget_master/-/blob/main/src/nuget-master/src/python-scripts/model/nuget_master.ipynb?ref_type=heads#:~:text=%23%20Save%20the%20model%20to%20a%20file)
2) Then place saved model file in a  directory _nuget_master/src/nuget-master/src/python-scripts/model/trained_models_
3) Set "auth" variable **personal access tokens (classic)**. Or you can remove it, but it may lead to GitHub rate limits after a while
4) Now, change working directory for ./src/nuget-master and run the project

https://github.com/razrez/nuget-master/assets/70781439/7c8ff54b-992e-4fce-8874-c382fde70ffd

# Плагин Visual Studio Code для рекомендации NuGet зависимостей на основе машинного обучения
#### В данной работе рассматривается разработка плагина, предназначенного для поиска рекомендаций NuGet зависимостей по текстовому описанию, которое вводится пользователем 
#### Актуальность темы заключается в том, что во время разработки ПО приходится тратить значимую часть времени на поиск и установку необходимых библиотек в проект, что отвлекает от основной задачи и мешает сосредоточиться.
#### Целью работы является разработка плагина для Visual Studio Code, предназначенного для рекомендации NuGet зависимостей, чтобы избавить разработчика от рутинных действий по установке и поиску NuGet зависимостей.
#### ```В ходе работы были изучены алгоритмы обработки текстовых данных, освоены методы разработки плагинов для Visual Studio Code​, осуществлены сбор, обработка  и структурирование данных, обучены модели векторизации и машинного поиска с применением FastText и KNN алгоритмов.```

____
Модель берёт текстовое описание (проекта/запроса), превращает его в вектор (с помощью FastText), а затем с помощью kNN находит наиболее близкие по смыслу репозитории из датасета. Идея в том, чтобы по текстовому описанию рекомендовать NuGet-зависимости, подходящие под задачи проекта. Сначала идёт сбор/очистка данных (убираем эмоджи, «мусор», приводим к единому формату), потом обучение модели векторизации (FastText), а далее построение kNN для быстрых рекомендаций по схожести текстовых описаний.
