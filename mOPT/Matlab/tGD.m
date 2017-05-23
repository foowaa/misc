function [x,y,varargout]=tGD(f,x0,nmax,epsilon,varargin)
%using GD to optimize
%x,y are output
%f is the function, g is the gradient function
%nmax is the maximum loops to accelerate the algorithm, epsilon is
%convergence condition
%step can choose Wolfe or Armijo conditions, Armijo is recomended

tic;
n=0;

while(n<nmax)
    grad=tautoGrad(f,x0,varargin{:});
    d=-grad; 
    if(norm(d)<epsilon), break; end
    alpha=tArmijo(f,grad,x0,d,varargin{:});
    x0=x0+alpha*d;
    n=n+1;
end
x=x0;
y=f(x0,varargin{:});
t=toc;
varargout{1}=t;
varargout{2}=n+1;