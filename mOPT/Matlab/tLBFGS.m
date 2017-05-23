function [x,y,varargout]=tLBFGS(f,x0,nmax,epsilon,corrections,varargin)
%using damped LBFGS method to optimize
%x,y are output
%f is the objective function
%nmax is the maximum loops to accelerate the algorithm, epsilon is
%convergence condition
%step can choose Wolfe or Armijo conditions, Wolfe is recomended.

tic;
n=0; 

while(n<nmax)
    if n==0
        grad=tautoGrad(f,x0,varargin{:});
        d=-grad;
        alpha=tWolfe(f,grad,x0,d,varargin{:});
        old_dirs = zeros(length(grad),0);
        old_stps = zeros(length(d),0);
        Hdiag = 1;
        grad_old=0;
    else
        if(norm(d)<epsilon), break; end
        grad=tautoGrad(f,x0,varargin{:});
        alpha=tWolfe(f,grad,x0,d,varargin{:});
        [old_dirs,old_stps,Hdiag] = lbfgsUpdate(grad-grad_old,alpha*d,corrections,old_dirs,old_stps,Hdiag);
        d = lbfgs(-grad,old_dirs,old_stps,Hdiag);
        grad_old=grad;
    end
    x0=x0+alpha*d;
    n=n+1;
end
x=x0;
y=f(x0,varargin{:});
t=toc;
varargout{1}=t;
varargout{2}=n+1;
end


function [old_dirs,old_stps,Hdiag] = lbfgsUpdate(y,s,corrections,old_dirs,old_stps,Hdiag)
S = old_dirs(:,2:end);
Y = old_stps(:,2:end);
k = size(Y,2);
L = zeros(k);
for j = 1:k
    for i = j+1:k
        L(i,j) = S(:,i)'*Y(:,j);
    end
end
D = diag(diag(S'*Y));
N = [S/Hdiag Y];
M = [S'*S/Hdiag L;L' -D];

ys = y'*s;
%M=M+modt*eye(length(M));%µ÷Õû
Bs = s/Hdiag - N*(M\(N'*s)); 
sBs = s'*Bs;

eta = .02;
if ys < eta*sBs
    theta = min(max(0,((1-eta)*sBs)/(sBs - ys)),1);
    y = theta*y + (1-theta)*Bs;
end

numCorrections = size(old_dirs,2);
if numCorrections < corrections
    old_dirs(:,numCorrections+1) = s;
    old_stps(:,numCorrections+1) = y;
else
    old_dirs = [old_dirs(:,2:corrections) s];
    old_stps = [old_stps(:,2:corrections) y];
end
Hdiag = (y'*s)/(y'*y);
end


function [d] = lbfgs(g,s,y,Hdiag)
[p,k] = size(s);
for i = 1:k
    ro(i,1) = 1/(y(:,i)'*s(:,i));
end
q = zeros(p,k+1);
r = zeros(p,k+1);
al =zeros(k,1);
be =zeros(k,1);
q(:,k+1) = g;
for i = k:-1:1
    al(i) = ro(i)*s(:,i)'*q(:,i+1);
    q(:,i) = q(:,i+1)-al(i)*y(:,i);
end
r(:,1) = Hdiag*q(:,1);

for i = 1:k
    be(i) = ro(i)*y(:,i)'*r(:,i);
    r(:,i+1) = r(:,i) + s(:,i)*(al(i)-be(i));
end
d=r(:,k+1);
end
