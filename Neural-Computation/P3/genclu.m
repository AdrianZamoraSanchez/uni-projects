function v = genclu(x,c,n,d)
% Funcion para generar clusters.
%  
% genclu(X,C,N,D)
%   X - matriz con los rangos de cada dimensión.
%   C - número de clusters a generar.
%   N - número de datos en cada cluster.
%   D - Desviación estándar de los clusters (valro por defecto: 1).
%
% Devuelve la matriz de datos generados según los valores indicados para
% los parámetros.

if nargin < 3, error(message('Deben proporcionarse al menos 3 argumentos')), end
% Si no se establece un valor para la desviación estándar, se fija el valor
% por defecto.
if nargin == 3, d = 1; end

[r,q] = size(x);
minv = min(x')';
maxv = max(x')';
v = rand(r,c) .* ((maxv-minv) * ones(1,c)) + (minv * ones(1,c));
t = c*n;
v = repmat(v,1,n) + randn(r,t)*d;