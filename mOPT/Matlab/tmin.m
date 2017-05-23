function [x,y,varargout]=tmin(f,x0,options,varargin)
%INPUT:
%f is the function-handle, x0 is initial value, options is a struct,
%varargin is other informations that f needs
%OUTPUT:
%x is a vector, y is a scaler, varargout is other imformations, included time,
%iteration times 
%OPTIONS:
%method:GD(gradient descent),NT(modified damped Newton), CG(conjugate gradient)
%,BFGS,LBFGS(damped Limited memory BFGS). LBFGS is default
%elpsilon: termination condition, default:1e-5
%nmax: termination condition, default:5000
%corrections: a parameter in LBFGS, the larger it is, the shorter time it 
%utilizes, default:100
warning('off','MATLAB:nearlySingularMatrix');
[method, epsilon, nmax, corrections]...
    =tProcessIn(options);

if strcmp(method,'GD')
    if nargout<3
        [x,y]=tGD(f,x0,nmax,epsilon,varargin{:});
    elseif nargout==3
        [x,y,varargout{1}]=tGD(f,x0,nmax,epsilon,varargin{:});
    else
        [x,y,varargout{:}]=tGD(f,x0,nmax,epsilon,varargin{:});
    end
elseif strcmp(method,'NT')
     if nargout<3
        [x,y]=tNT(f,x0,nmax,epsilon,varargin{:});
    elseif nargout==3
        [x,y,varargout{1}]=tNT(f,x0,nmax,epsilon,varargin{:});
    else
        [x,y,varargout{:}]=tNT(f,x0,nmax,epsilon,varargin{:});
     end   
elseif strcmp(method,'CG')
     if nargout<3
        [x,y]=tCG(f,x0,nmax,epsilon,varargin{:});
    elseif nargout==3
        [x,y,varargout{1}]=tCG(f,x0,nmax,epsilon,varargin{:});
    else
        [x,y,varargout{1},varargout{2}]=tCG(f,x0,nmax,epsilon,varargin{:});
     end   
elseif strcmp(method,'BFGS')
     if nargout<3
        [x,y]=tBFGS(f,x0,nmax,epsilon,varargin{:});
    elseif nargout==3
        [x,y,varargout{1}]=tBFGS(f,x0,nmax,epsilon,varargin{:});
    else
        [x,y,varargout{:}]=tBFGS(f,x0,nmax,epsilon,varargin{:});
     end   
elseif strcmp(method,'LBFGS')
     if nargout<3
        [x,y]=tLBFGS(f,x0,nmax,epsilon,corrections,varargin{:});
    elseif nargout==3
        [x,y,varargout{1}]=tLBFGS(f,x0,nmax,epsilon,corrections,varargin{:});
    else
        [x,y,varargout{1},varargout{2}]=tLBFGS(f,x0,nmax,epsilon,corrections,varargin{:});
     end
end
    end   