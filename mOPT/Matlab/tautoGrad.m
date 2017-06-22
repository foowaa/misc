function g = autoGrad(funObj,x,varargin)
% [f,g] = autoGrad(x,useComplex,funObj,varargin)
% funObj is the objective function, x is initial value
% Numerically compute gradient of objective function from function values
%
% type =
%     1 - forward-differencing (p+1 evaluations)
%     2 - central-differencing (more accurate, but requires 2p evaluations)

% default type = 2

type=2;

p = length(x);
if type == 1 % Use Finite Differencing
	f = funObj(x,varargin{:});
	mu = 2*sqrt(1e-12)*(1+norm(x));
	diff = zeros(p,1);
	for j = 1:p
		e_j = zeros(p,1);
		e_j(j) = 1;
		diff(j,1) = funObj(x + mu*e_j,varargin{:});
	end
	g = (diff-repmat(f,p,1))/mu;

else % Use Central Differencing
	mu = 2*sqrt(1e-12)*(1+norm(x));
	diff1 = zeros(p,1);
	diff2 = zeros(p,1);
	for j = 1:p
		e_j = zeros(p,1);
		e_j(j) = 1;
		diff1(j,1) = funObj(x + mu*e_j,varargin{:});
		diff2(j,1) = funObj(x - mu*e_j,varargin{:});
    end
	g = (diff1 - diff2)/(2*mu);
end
