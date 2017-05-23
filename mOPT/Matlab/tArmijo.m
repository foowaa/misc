function alpha=tArmijo(f,grad,xk,dk,varargin)
%Armijo condition for choosing a step
%alpha is the scaler
%f is the function, g is the gradient, xk is the value, dk is the direction
%beta and sigma is parameters, beta belongs (0,1),  sigma belongs (0,0.5)
%nmax is the maximum loops to accelerate the algorithm and controls alpha

beta=0.5; sigma=0.2;
n=0;smax=20;

while(n<smax)
    if (f(xk+beta^n*dk,varargin{:}))<=f(xk,varargin{:})+sigma*beta^n*grad'*dk; break,end
    n=n+1;
end
alpha=beta^n;