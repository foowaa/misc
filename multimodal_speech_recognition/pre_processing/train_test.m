clear all
clc
close all


v = h5read('video_cnn.h5','/cnn');
a = h5read('data.h5','/audio');
l = h5read('data.h5','/label');

% for j=1:6
%     f = l(1+3600*(j-1):3600*j,:);
%     unique(f)
% end
meanv = repmat(mean(v),21600,1);
meana = repmat(mean(a),21600,1);
%meanl = repmat(mean(l),21600,1);

v = v-meanv;
a = a-meana;
%l = l-meanl;

rng('shuffle');
r = randi([0,8], [1,60]);
% r = load('rr.mat','r');
% r=r.r;


vf = zeros(2400,100);
af = zeros(2400,200);
lf = zeros(2400,1);
x = zeros(60,1);
y = x;


for i = 0:59
    rt = r(i+1);
    x(i+1) = rt*40+1+360*i;
    y(i+1) = (rt+1)*40+360*i;
    vt = v(x(i+1):y(i+1),:);
    at = a(x(i+1):y(i+1),:);
    lt = l(x(i+1):y(i+1),:);
    
    p = i*40+1;
    vf(i*40+1:(i+1)*40,:) = vt;
    af(i*40+1:(i+1)*40,:) = at;
    lf(i*40+1:(i+1)*40,:) = lt;
    
end
    
for j = 60:-1:1
    v(x(j):y(j),:) = [];
    a(x(j):y(j),:) = [];
    l(x(j):y(j),:) = [];
end
    
    
    