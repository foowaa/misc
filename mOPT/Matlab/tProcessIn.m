function [method, epsilon, nmax, corrections]...
    =tProcessIn(o)
o = toUpper(o);

nmax=getOption(o,'NMAX',5000);
epsilon=getOption(o,'EPSILON',1e-5);
corrections=getOption(o,'CORRECTIONS',100);
method=getOption(o,'METHOD','LBFGS');

end

function [v] = getOption(options,opt,default)
if isfield(options,opt)
    if ~isempty(getfield(options,opt))
        v = getfield(options,opt);
    else
        v = default;
    end
else
    v = default;
end
end

function [o] = toUpper(o)
if ~isempty(o)
    fn = fieldnames(o);
    for i = 1:length(fn)
        o = setfield(o,upper(fn{i}),getfield(o,fn{i}));
    end
end
end