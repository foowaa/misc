function alpha = tWolfe(f,grad,xk,dk,varargin)
%Wolfe condition for choosing a step
%alpha is a scaler
%f is the function, grad is the gradient, xk is the value, dk is the direction
%c1 and c2 is parameters, c1 belongs (0,1), usually very small, c2 belongs
%(c1,1)
%nmax is the maximum loops to accelerate the algorithm

c1 = 0.1; c2 = 0.9;
alpha = 1; a = 0; b = Inf; n=0;smax=20;

while (n<smax)
    if f(xk+alpha*dk,varargin{:})>f(xk,varargin{:})+c1*alpha*grad'*dk
        b = alpha;
        alpha = (alpha+a)/2;
        n=n+1;
        continue;
    elseif autoGrad(f,xk+alpha*dk,varargin{:})'*dk<c2*grad'*dk
        a = alpha;
        alpha = min([2*alpha, (b+alpha)/2]);
        n=n+1;
        continue;
    else
        break;
    end  
end
if alpha>1
    rng('shuffle');
    alpha=0.01 + 0.09.*rand(1,1);
end
end
