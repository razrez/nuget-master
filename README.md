# Nuget Master
1) Firstly, save the FastText model like this [here](https://git.kpfu.ru/RIKhodzhamirzoev/nuget_master/-/blob/main/src/nuget-master/src/python-scripts/model/nuget_master.ipynb?ref_type=heads#:~:text=%23%20Save%20the%20model%20to%20a%20file)
2) Then place saved model file in a  directory _nuget_master/src/nuget-master/src/python-scripts/model/trained_models_
3) Set "auth" variable **personal access tokens (classic)**. Or you can remove it, but it may lead to GitHub rate limits after a while
4) Now, change working directory for ./src/nuget-master and run the project

![image](https://github.com/razrez/nuget-master/assets/70781439/b673282c-0481-48c2-960e-1ce532dd3f43)

![image](https://github.com/razrez/nuget-master/assets/70781439/5ec40de6-7474-471b-ac01-0823264d23fc)

![image](https://github.com/razrez/nuget-master/assets/70781439/4082942e-03f6-425e-aa6e-051f5b462022)


# плагин Visual Studio Code для рекомендации NuGet зависимостей на основе машинного обучения
#### В данной работе рассматривается разработка плагина, предназначенного для рекомендации NuGet зависимостей по текстовому описанию разрабатываемого проекта.
#### Актуальность темы заключается в том, что во время разработки ПО приходится тратить значимую часть времени на поиск и установку необходимых библиотек в проект, что отвлекает от основной задачи и мешает сосредоточиться.
#### Целью работы является разработка плагина для Visual Studio Code, предназначенного для рекомендации NuGet зависимостей, чтобы избавить разработчика от рутинных действий по установке и поиску NuGet зависимостей.
#### Для достижения цели были поставлены следующие задачи:
*	Изучить основные алгоритмы обработки текстовых данных с применением машинного обучения и методы разработки плагинов для Visual Studio Code
*	Составить методологию по реализации проекта
*	Осуществить сбор данных из открытых источников
*	Реализация модели машинного поиска и её тестирование
*	Создание плагина для Visual Studio Code с использованием разработанной модели машинного поиска
