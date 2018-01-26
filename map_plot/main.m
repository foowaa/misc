clear all;
close all;

%load practice
load('bin.mat');
oxts = loadOxtsliteData('E:\programming\dataSet\2011_09_26_drive_0002_sync\2011_09_26\2011_09_26_drive_0002_sync');
lat = zeros(77,1);
lon = zeros(77,1);
X = zeros(77,1);
Y = zeros(77,1);
N = 2000;
N1 = 2000;
height = zeros(N1, 77);
theta = zeros(77, 1);
ranges = zeros(N, 77);
rate = 0.001*0.5;
for i=1:77
    lat(i) = oxts{i}(1);
    lon(i) = oxts{i}(2);
    [X(i),Y(i)] = millerXY (lon(i), lat(i));
    theta(i) = oxts{i}(6);
    v = filecell{i}(1:3,:);
    v = v(:,1:N);
    v = v(1,:).*v(1,:)+v(2,:).*v(2,:);
    ranges(:,i) = v.^0.5*rate;
    vh = filecell{i}(3,:);
    vh = vh(:,1:N1);
    %height(:,i) = vh*rate;
    height(:, i) = round(vh);
end

X1 = X(1:76);
X2 = X(2:77);
Y1 = Y(1:76);
Y2 = Y(2:77);
deltaX = X2 - X1;
deltaY = Y2 - Y1;

locx = zeros(77,1);
locy = zeros(77,1);
locx(1) = 0;
locy(1) = 0;
for i=2:77
    locx(i) = (locx(i-1)+deltaX(i-1));
    locy(i) = (locy(i-1)+deltaY(i-1));
end

%theta = zeros(77,1);
pose = [-locx'*rate*10; locy'*rate*10; theta'];

scanAngles = zeros(N, 1);
stepAngles = 6/(N-1);
scanAnglesTemp = 3:-stepAngles:-3;
scanAngles(:,1) = scanAnglesTemp;
% This will load four variables: ranges, scanAngles, t, pose
% [1] t is K-by-1 array containing time in second. (K=3701)
%     You may not use time info for implementation.
% [2] ranges is 1081-by-K lidar sensor readings. 
%     e.g. ranges(:,k) is the lidar measurement (in meter) at time index k.
% [3] scanAngles is 1081-by-1 array containing at what angles (in radian) the 1081-by-1 lidar
%     values ranges(:,k) were measured. This holds for any time index k. The
%     angles are with respect to the body coordinate frame.
% [4] pose is 3-by-K array containing the pose of the mobile robot over time. 
%     e.g. pose(:,k) is the [x(meter),y(meter),theta(in radian)] at time index k.

% 1. Decide map resolution, i.e., the number of grids for 1 meter.
param.resol = 50;

% 2. Decide the initial map size in pixels
param.size = [700*rate*100, 1000*rate*100];

% 3. Indicate where you will put the origin in pixels
param.origin = [200*rate*100,300*rate*100]'; 

% 4. Log-odd parameters 
param.lo_occ = 1;
param.lo_free = 0.5; 
param.lo_max = 100;
param.lo_min = -100;


% 5. Recod video Bool
%load 'ws1.mat';
param.captureBool = 1;

% Mapping function
myMap = occGridMapping(ranges, scanAngles, pose, height, param);

% The final grid map: 
figure(2),
imagesc(myMap);
colormap('gray'); axis equal;