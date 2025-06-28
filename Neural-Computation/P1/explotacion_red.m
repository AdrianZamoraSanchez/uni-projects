% Importa la red deseada (la más pequeña)
load('red_neuronal_3_neuronas.mat', 'net');

% Nuevos datos generados por chatgpt
test_data = readtable('bodyfat_explotacion.csv');

% Características: todas las columnas excepto IDNO y BODYFAT
X_test = test_data{:, 3:end}';

% Valores reales de BODYFAT para comparar
Y_test = test_data{:, 2}';      

Y_pred = net(X_test);

% Mostrar predicciones realizadas y valores reales
disp('Predicciones:');
disp(Y_pred);
disp('Valores reales:');
disp(Y_test);

% Calcula el MSE total y lo muestra en pantalla
mse_error = mse(Y_test - Y_pred);
fprintf('MSE en los datos de prueba: %.2f%%\n', mse_error);