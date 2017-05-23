function [x,y,varargout]=tNT(f,x0,nmax,epsilon,varargin)
%using modified damped Newton method to optimize
%x,y are output
%f is the objective function
%nmax is the maximum loops to accelerate the algorithm, epsilon is
%convergence condition
%step can choose Wolfe or Armijo conditions, Wolfe is recomended
%healthy H is fit with Newton methods, otherwise it's time-wasteing and
%usually wrong solution
%rho is parameter, which controls the strength of modification 
%[x,y]=tNewton(@f,[0;0])
tic;
n=0; rho=0.0;

while(n<nmax)
    grad=autoGrad(f,x0,varargin{:});
    muk=norm(grad)^(1+rho);
    hes=autoHess(f,x0,varargin{:});
    hesm=hes+muk*eye(length(x0));
    d=-hesm\grad;
    if(norm(grad)<epsilon), break; end
    alpha=tWolfe(f,grad,x0,d,varargin{:});
    x0=x0+alpha*d;
    n=n+1;
end
x=x0;
y=f(x0,varargin{:});
t=toc;
varargout{1}=t;
varargout{2}=n+1;