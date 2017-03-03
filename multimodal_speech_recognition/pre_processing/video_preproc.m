%This file is to preprocess the video frame
%@Author: cltian
%@Date: 2016-10-10


clear all
clc
close all
%folder = uigetfile('*.avi');
cd F:\database\AVD\video\
%TODO cannot find files
n1 = 'S4';
my_cell1 = cell(1,50); my_cell2 = cell(1,50);
cell1i = 1; cell2i = 1;
for n2 = 0:9
    for n3 = 1:9
        fn = strcat(n1, '_',num2str(n2),'_','0',num2str(n3),'.avi');
        obj = VideoReader(fn);
        v = read(obj);
        size1 = size(v);
        num_frames = size1(4);
%There are 6 different people here, everybody need distinct params.
%1st : 'MinSize',[200,500],'MaxSize',[400,900]
%2nd : 'MinSize',[150,250],'MaxSize',[170,300]
%3rd : 'MinSize',[150,250],'MaxSize',[170,280], choose the smaller one
%bbox(1,1) vs bbox(2,1)
%4th : 'MinSize',[150,270],'MaxSize',[170,300]
%5th : 'MinSize',[270,500] 小的那个
%6th : MinSize',[200,400], 'MaxSize',[300,450]
%i=1;
          for i=1:num_frames
             I=rgb2gray(v(:,:,:,i));
             mouthDetector = vision.CascadeObjectDetector('ClassificationModel','Mouth', 'MinSize',[150,270],'MaxSize',[170,300]);
             bbox = step(mouthDetector,I);
              %Debug 
%            IMouth = insertObjectAnnotation(I, 'rectangle', bbox, 'Mouth');
%            figure, imshow(IMouth), title(strcat(path,n1, '_',num2str(n2),'_','0',num2str(n3),'_',num2str(i)));
             size_temp = size(bbox);
             path = 'F:\database\AVD\pic\pic\';
             fileName = strcat(path,n1, '_',num2str(n2),'_','0',num2str(n3),'_',num2str(i),'.bmp');
            if(size_temp(1)==1)
                  I1 = imcrop(I,bbox);
                  imwrite(I1, fileName, 'bmp');
            elseif(size_temp(1)==0)
                   cell_str1 = strcat(n1, '_',num2str(n2),'_','0',num2str(n3),'_',num2str(i));
                   my_cell1{celli} = cell_str;
                   cell1i=cell1i+1;
                   %assert(celli<25, 'too many empty frames');
            elseif(size_temp(1)==2)
                  [~,index] = min(bbox(:,1));
                  %bbox(index,:)=[];
                  bbox_temp = bbox(index,:);
                  I1 = imcrop(I,bbox_temp);
                  imwrite(I1, fileName, 'bmp');
            %assert(size_temp(1)==1, 'Frame:%d',i)
%            elseif(size_temp(1)==3)
%                 [~,index1]=min(bbox(:,1));
%                 bbox(index1,:)=[];
%                 [~,index2]=max(bbox(:,1));
%                 bbox(index2,:)=[];
%                 [~,index]=max(bbox(:,2));
%                 bbox=bbox(index,:);
%                 I1=imcrop(I,bbox);
%                 imwrite(I1,fileName,'bmp');
            else
                cell_str2 = strcat(n1, '_',num2str(n2),'_','0',num2str(n3),'_',num2str(i));
                 my_cell2{celli} = cell_str;
                 cell2i=cell2i+1;               
            end
       end
   end
end
save my_cell1
save my_cell2


