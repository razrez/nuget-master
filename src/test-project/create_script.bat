@echo off

:: Create a new solution
dotnet new sln -n MySolution

:: Create a new Web API project
dotnet new webapi -n MyWebApi
dotnet sln MySolution.sln add MyWebApi/MyWebApi.csproj

:: Create a new Class Library project
dotnet new classlib -n MyLib1
dotnet sln MySolution.sln add MyLib1/MyLib1.csproj

:: Create another new Class Library project
dotnet new classlib -n MyLib2
dotnet sln MySolution.sln add MyLib2/MyLib2.csproj