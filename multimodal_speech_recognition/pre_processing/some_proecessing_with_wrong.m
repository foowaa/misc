clc 
clear all

cd F:\database\AVD\video\



mouthDetector = vision.CascadeObjectDetector('ClassificationModel','Mouth', 'MinSize',[150,250],'MaxSize',[170,300]);
path = 'F:\database\AVD\pic\pic\pic\';
threshold = 50;
fn = 'S2_0_01.avi';frame=44;
obj = VideoReader(fn);
    v = read(obj, frame);
    I = rgb2gray(v);
    bbox = step(mouthDetector,I);
        IMouth = insertObjectAnnotation(I, 'rectangle', bbox, 'Mouth');
    figure, imshow(IMouth);
%     [~,index]=max(bbox(:,2));
%     bbox=bbox(index,:);
%     I1=imcrop(I,bbox);
%     imwrite(I1,fileName,'bmp');
%     figure, imshow(I1);
    sorted_bbox_x = sortrows(bbox,1);
    %sorted_bbox_y = sortrows(bbox,2);
    if(sorted_bbox_x(3,1)-sorted_bbox_x(2,1)<threshold)
        bbox_temp = bbox(2:end,:);
        [~,index] = min(bbox_temp(:,2));
        bbox_temp = bbox_temp(index, :);
    else
        bbox_temp = bbox;
        [~,index] = max(bbox_temp(:,2));
        bbox_temp = bbox_temp(index,:);
    end
    IMouth = insertObjectAnnotation(I, 'rectangle', bbox_temp, 'Mouth');
    figure, imshow(IMouth)
