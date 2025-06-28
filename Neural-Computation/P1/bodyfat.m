% Se cargan los datos del archivo CSV que contienen medidas corporales y el 
% porcentaje de grasa corporal de los individuos.
data = readtable('bodyfat.csv');

% Se separan las características (variables predictoras) en X y el objetivo 
% a predecir (BODYFAT) en Y. Se usa la transposición (') para que las columnas 
% de los datos se conviertan en vectores adecuados para la red neuronal
X = data{:, 3:end}';  % Características: todas las columnas excepto IDNO y BODYFAT
Y = data{:, 2}';      % Objetivo: columna BODYFAT

% Se definen las configuraciones de número de neuronas a probar
neurons_to_test = [3, 5, 15, 60];  % Vector con Nº de neuronas a probar
results = zeros(length(neurons_to_test), 1);  % Vector para almacenar el MSE de cada configuración
train_times = zeros(length(neurons_to_test), 1);  % Vector para almacenar el tiempo de entrenamiento

% Límite del error en la prediccón
error_limite = 2.5;

% Bucle para entrenar la red con diferentes cantidades de neuronas
for i = 1:length(neurons_to_test)
    % Se selecciona la cantidad de neuronas para esta iteración
    num_neurons = neurons_to_test(i);
    
    % Función de entrenamiento
    trainFcn = 'trainlm';
    
    % Crear y configurar la red neuronal
    net = fitnet(num_neurons, trainFcn);
    
    % División aleatoria de los datos
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    
    % Se inicia el cronormetro, se entrena la red y se para el cronometro,
    % obteniendo el tiempo de entrenamiento
    tic;
    [net, tr] = train(net, X, Y); % Se entrena la red
    train_time = toc;

    % Almacena el valor del tiempo de entrenamiento
    train_times(i) = train_time;
    
    % Obtener los índices del conjunto de prueba
    testInd = tr.testInd;

    % Predecir en el conjunto de prueba
    Y_pred = net(X(:, testInd));
    Y_test = Y(testInd);
    
    % Calcular el % de error absoluto
    abs_error = abs(Y_test - Y_pred);
    
    % Contar el número de aciertos, es decir todos los valores con error absoluto por debajo del 2%
    aciertos = sum(abs_error < error_limite);

    % Contar el número de aciertos, es decir todos los valores con error absoluto por encima del 2% 
    fallos = sum(abs_error >= error_limite);
    
    % Calcular el MSE y lo almacena
    mse_error = mse(Y_test - Y_pred);
    results(i) = mse_error;
    
    % Se muestran los datos y resultados para esta red neuronal
    fprintf('Con %d neuronas: MSE: %.4f, Aciertos: %d, Fallos: %d, Fallo del %.2f%%, Tiempo de entrenamiento: %.2f segundos\n', ...
        num_neurons, mse_error, aciertos, fallos, (fallos/(aciertos+fallos))*100, train_time);

    % Pausar la ejecución hasta que se presione Enter para ver los resultados del entrenamiento
    % input('Presiona Enter para continuar...');

    % Codigo con el que se exporta la red deseada
    filename = sprintf('red_neuronal_%d_neuronas.mat', num_neurons);
    save(filename, 'net');
    
end

% Mostrar el resumen de los resultados del MSE y el tiempo de entrenamiento
disp('Resultados de MSE para cada configuración de neuronas:');
disp(results);
disp('Tiempos de entrenamiento para cada configuración de neuronas:');
disp(train_times);
