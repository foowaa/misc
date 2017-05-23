function [x,y,varargout]=tBFGS(f,x0,nmax,epsilon,varargin)
%using BFGS quasi-Newton method to optimize
%x,y are output
%f is the objective function
%nmax is the maximum loops to accelerate the algorithm, epsilon is
%convergence condition
%step can choose Wolfe or Armijo conditions, Wolfe is recomended
%set B0=E
tic;
n=0; 
B=eye(length(x0));

x=zeros(length(x0));
while(n<nmax)
    grad=tautoGrad(f,x0,varargin{:});
    d=-B\grad;
    if(norm(d)<epsilon), break; end
    alpha=tWolfe(f,grad,x0,d,varargin{:});
    x=x0+alpha*d;
    s=x-x0;
    yk=autotGrad(f,x,varargin{:})-grad;
    if yk'*s>0
        B=B-(B*s*s'*B)/(s'*B*s)+(yk*yk')/(yk'*s);
    end
    n=n+1;
    x0=x;
end
y=f(x0,varargin{:});
t=toc;
varargout{1}=t;
varargout{2}=n+1;