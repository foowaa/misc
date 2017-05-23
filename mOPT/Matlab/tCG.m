function [x,y,varargout]=tCG(f,x0,nmax,epsilon,varargin)
%using Fletcher-Reeves CG method to optimize nonlinear problems
%x,y are output
%f is the function
%nmax is the maximum loops to accelerate the algorithm, epsilon is
%convergence condition
%step can choose Wolfe or Armijo conditions, Wolfe is recomended

tic; 
n=0; 

grad=tautoGrad(f,x0,varargin{:});
d=-grad;
while(n<nmax)
    if(norm(d)<epsilon), break; end
    alpha=tWolfe(f,grad,x0,d,varargin{:});
    x0=x0+alpha*d;
    temp=tautoGrad(f,x0,varargin{:});
    beta=temp'*temp/(grad'*grad);
    d=-temp+beta*d;
    n=n+1;
    grad=temp;
end
x=x0;
y=f(x0,varargin{:});
t=toc;
varargout{1}=t;
varargout{2}=n+1;