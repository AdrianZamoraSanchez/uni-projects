%% Práctica Nº2, por: Arián Zamora Sánchez
% Esta práctica consiste en el análisis sobre el ajuste a una función
% de redes MLP
%

% Para tener limpio el entorno de pruebas al hacer muchas ejecuciones
clear; clc;

% Precisión objetivo de las redes
goal = 1e-5;

% Neuronas con las que se hace la prueba de ambas redes
n_neurons = [5, 10, 50, 100, 500];

% Generar los datos en el intervalo [-6, 6]
x = linspace(-6, 6, 1000); % 1000 puntos entre -6 y 6
y = 0.2*x + sqrt(sqrt(abs(x))); % Función objetivo

% Dividir los datos en entrenamiento (70%), validación (15%) y prueba (15%)
[trainInd, valInd, testInd] = dividerand(1000, 0.7, 0.15, 0.15);

for i = 1:length(n_neurons)
    %--- Pruebas para la red MLP ---%
    % Se crea y configura la red Perceptrón Multicapa (MLP)
    net = feedforwardnet(n_neurons(i), 'trainlm');
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInd;
    net.divideParam.valInd = valInd;
    net.divideParam.testInd = testInd;
    net.trainParam.epochs = 30;
    net.trainParam.goal = goal;
    
    % Entrenar el MLP
    start_mlp = cputime; % Inicio del temporizador de entrenamiento
    [net, tr] = train(net, x, y);
    end_mlp = cputime - start_mlp; % Inicio del temporizador de entremianto

    % Predecir y evaluar el rendimiento
    test_mlp = net(x(testInd));
    
    % Calcular el error MSE en el conjunto de prueba
    mse_mlp = mse(net, y(testInd), test_mlp);
    fprintf('MSE - Tiempo(s) de la red MPL: %f - %f con %d neuronas\n', mse_mlp, end_mlp, n_neurons(i));
    
    %--- Pruebas para la red RBF ---%
    % Crear y entrenar una Red de Función de Base Radial (RBF)
    start_rbf = cputime; % Inicio del temporizador de entrenamiento
    % Se utiliza evalc con la función newrb como argumento para ignorar la
    % salida de la consola (~) pero mantener la red almacenada en net_ebf
    [~, net_rbf] = evalc('newrb(x(trainInd), y(trainInd), goal, 1, n_neurons(i))');
    end_rbf = cputime - start_rbf; % Inicio del temporizador de entremianto
    
    % Predecir y evaluar el rendimiento
    test_rbf = net_rbf(x(testInd));
    
    % Calcular el error MSE en el conjunto de prueba
    mse_rbf = mse(net_rbf, y(testInd), test_rbf);
    fprintf('MSE - Tiempo(s) de la red RBF: %f - %f con %d neuronas\n', mse_rbf, end_rbf, n_neurons(i));
    
    %--- Visualización de los resultados ---%
    figure;
    plot(x, y, 'b', 'LineWidth', 1.5); hold on;
    plot(x(testInd), test_mlp, 'g--', 'LineWidth', 1.5);
    plot(x(testInd), test_rbf, 'r--', 'LineWidth', 1.5);
    title(strcat('Comparación de las redes con:  ', num2str(n_neurons(i)), ' neuronas'));
    legend('Datos reales', 'Predicción de la MPL', 'Prediccón de la RBF');
    xlabel('x'); ylabel('f(x)');
    grid on;
end