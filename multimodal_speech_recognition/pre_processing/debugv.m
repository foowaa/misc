%This file detect problems in video_processing.m
%Due to the selection of parameters, Viola_Jones algorithm usually
%incorrect. Therefore, if candidates are more than 2 OR less than 1, its
%number is recorded in my_cell2 and my_cell1. 
%Here I sample in my_cell*, then using figures to confirm the reasons.
clc 
clear all
close all
cd F:\database\AVD\video\
load my_cell
length1 = 7;
length2 = length(my_cell);
r = randi([1 length2],20,1);
for i=1:20
    str = my_cell{i};
    str1 = str(1:7);
    str2 = str(9:end);
    fn = strcat(str1,'.avi');
    frame = str2num(str2);
    obj = VideoReader(fn);
    v = read(obj, frame);
    I = rgb2gray(v);
    mouthDetector = vision.CascadeObjectDetector('ClassificationModel','Mouth','MinSize',[150,250],'MaxSize',[170,280]);
    bbox = step(mouthDetector,I);
    [~,index]=max(bbox(:,2));
    bbox=bbox(index,:);
    I1=imcrop(I,bbox);
    figure, imshow(I1);
%     IMouth = insertObjectAnnotation(I, 'rectangle', bbox, 'Mouth');
%     figure, imshow(IMouth), title(my_cell{i});
end