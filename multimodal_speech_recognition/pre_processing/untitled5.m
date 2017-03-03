clear all,clc
video = h5read('data_train.h5','/video_train');
audio = h5read('data_train.h5','/audio_train');
label = h5read('data_train.h5','/label_train');


rng('shuffle');
noisev = normrnd(0,2,19200,100);
noisea = normrnd(0,2,19200,200);
video = video+video;
audio = audio+audio;
ran = randi([-10,10],480,1);
x1 = zeros(19200,100);
x2 = zeros(19200,200);
x = cell(5,1);
y = cell(5,1);

for j=1:1
x1 = zeros(19200,100);
x2 = zeros(19200,200);
for i = 1:480
    index = (i-1)*40+1;
    if ran(i)<0
       rantemp = -ran(i);
       x1(index:index+rantemp-1,:) = video(i*40-rantemp+1:i*40,:);
       x2(index:index+rantemp-1,:) = audio(i*40-rantemp+1:i*40,:);
       x1(index+rantemp:i*40,:) = video(index:i*40-rantemp,:);
       x2(index+rantemp:i*40,:) = audio(index:i*40-rantemp,:);
       
    elseif ran(i)>0
        rantemp = rand(i);
        x1(i*40-rantemp+1:i*40,:) = video(index:index+rantemp-1,:);
        x2(i*40-rantemp+1:i*40,:) = audio(index:index+rantemp-1,:);
        x1(index:i*40-rantemp,:) = video(index+rantemp:i*40,:);
        x2(index:i*40-rantemp,:) = audio(index+rantemp:i*40,:);
    else
        
    end
end
x{j} = x1;
y{j} = x2;
end
video_aug = [video;x{1}];
audio_aug = [audio;y{1}];
label_aug = [label;label];

h5create('data_aug3','/video_train',[19200*2,100]);
h5create('data_aug3','/audio_train',[19200*2,200]);
h5create('data_aug3','/label_train',[19200*2,1]);
h5write('data_aug3','/video_train',video_aug);
h5write('data_aug3','/audio_train',audio_aug);
h5write('data_aug3','/label_train',label_aug);