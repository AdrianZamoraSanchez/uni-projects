%% Práctica Nº3, por: Arián Zamora Sánchez
% Esta práctica consiste en el análisis de las cualidades
% de las redes una red SOM bidimensional.

% Para tener limpio el entorno de pruebas al hacer cada ejecución
clear; clc;


% DATOS
% Definir parámetros de genclu
X = [0 10; 0 10; 0 10];  % Rango de centros de cada dimensión
C = 4;  % Número de clusters
N = 100;  % Número de puntos en cada cluster
D = 0.3;  % Desviación estándar

% Generar los datos
data = genclu(X, C, N, D);

% Visualizar los datos generados (opcional)
figure;
scatter3(data(1,:), data(2,:), data(3,:));
title('Datos generados con genclu');
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;


% EXPERIMENTOS con diferentes configuraciones
% Definir posibles valores de parámetros
gridSizes = {[5 5], [10 10], [25,25]};
topologyFcns = {'hextop', 'gridtop'};
distanceFcns = {'dist', 'linkdist'};
neiborhoodSize = {1,3};

% Variables utilizadas para calcular el MSE
totalError = 0;
dataPoints = size(data, 2);

% Hacer un bucle para probar todas las combinaciones
for sizeIndex = 1:length(gridSizes)
    for topologyIndex = 1:length(topologyFcns)
        for distanceIndex = 1:length(distanceFcns)
            for neiborhoodSizeIndex = 1:length(neiborhoodSize)
                % Variables de experimentación
                size = gridSizes{sizeIndex};
                topologyFn = topologyFcns{topologyIndex};
                distanceFn = distanceFcns{distanceIndex};
                neiborhood = neiborhoodSize{neiborhoodSizeIndex};

                % Creación del SOM
                som = selforgmap(size, 100, neiborhood, topologyFn, distanceFn);

                % Entrenar el SOM con los datos generados
                tic; % Se inicia temporizador
                som = train(som, data);
                trainTime = toc; % Fin del temporizador

                % Se camputan los resultados
                y = som(data);
                winnerId = vec2ind(y);  % Índice de la neurona ganadora para cada dato
                
                % Obtener los pesos de las neuronas ganadoras
                weights = som.iw{1};  % Pesos de las neuronas
                
                % Inicializar una variable para acumular el error cuadrático
                totalError = 0;
                
                % Calcular el MSE
                for i = 1:dataPoints
                    % Extraer el dato original
                    dataPoint = data(:, i);
                    
                    % Obtener el prototipo de la neurona ganadora (peso de la neurona)
                    winningNeuronWeight = weights(winnerId(i), :)';
                    
                    % Calcular el error cuadrático para este dato
                    error = sum((dataPoint - winningNeuronWeight).^2);
                    
                    % Acumular el error
                    totalError = totalError + error;
                end
                
                % Calcular el MSE
                mse = totalError / dataPoints;
                
                % Mostrar el resultado del MSE
                disp(['MSE y tiempo para: ', 'Grid size ', num2str(size(1)), 'x', num2str(size(2)), ...
                    ', Neiborhood ', num2str(neiborhood), ', TopologyFun ', topologyFn, ...
                    ', DistanceFun ', distanceFn, ' es = ', num2str(mse), ' , ',num2str(trainTime)]);
            
                % Crear una carpeta con el nombre del tamaño de la rejilla
                folderName = ['GridSize_', num2str(size(1)), 'x', num2str(size(2))];
                if ~exist(folderName, 'dir')  % Si la carpeta no existe, se crea
                    mkdir(folderName);
                end
                
                % Crear el nombre de subcarpeta con la configuración actual (vecindario, topología, distancia)
                subFolderName = ['Neiborhood_', num2str(neiborhood), '-TopologyFun_', topologyFn, '-DistanceFun_', distanceFn];
                subFolderPath = fullfile(folderName, subFolderName);
                if ~exist(subFolderPath, 'dir')  % Si la subcarpeta no existe, se crea
                    mkdir(subFolderPath);
                end
                
                % --- Plotsomnd (Mapa de nodos del SOM) ---
                figure;
                plotsomnd(som);
                title(['Plotsomnd ', 'Grid size ', num2str(size(1)), 'x', num2str(size(2)), ', Neiborhood ', ...
                    num2str(neiborhood), ', TopologyFun ', topologyFn, ', DistanceFun ', distanceFn]); % Titulo de la figura
                filename = fullfile(subFolderPath, 'plotsomnd.png');  % Guardar la figura en la subcarpeta
                saveas(gcf, filename);
                close(gcf);  % Cerrar para no saturar
                
                % --- Plotsomhits (Mapa de activación de SOM) ---
                figure;
                plotsomhits(som, data);  % Muestra cuántas veces ha sido activada cada neurona
                title(['Plotsomhits ', 'Grid size ', num2str(size(1)), 'x', num2str(size(2)), ', Neiborhood ', ...
                    num2str(neiborhood), ', TopologyFun ', topologyFn, ', DistanceFun ', distanceFn]); % Titulo de la figura
                filename = fullfile(subFolderPath, 'plotsomhits.png');  % Guardar la figura en la subcarpeta
                saveas(gcf, filename);
                close(gcf);  % Cerrar para no saturar
                
                % --- Plotsompos (Posiciones de las neuronas respecto a los datos) ---
                figure;
                plotsompos(som, data);  % Muestra las posiciones de las neuronas
                title(['Plotsompos ', 'Grid size ', num2str(size(1)), 'x', num2str(size(2)), ', Neiborhood ', ...
                    num2str(neiborhood), ', TopologyFun ', topologyFn, ', DistanceFun ', distanceFn]); % Titulo de la figura
                filename = fullfile(subFolderPath, 'plotsompos.png');  % Guardar la figura en la subcarpeta
                saveas(gcf, filename);
                close(gcf);  % Cerrar para no saturar
            end
        end
    end
end