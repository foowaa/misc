% Occupancy Grid Mapping function
function myMap = occGridMapping(ranges, scanAngles, pose, height, param)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% Parameters 
% 
% the number of grids for 1 meter.
 myResol = param.resol;
% the initial map size in pixels
 myMap = zeros(param.size);
% the origin of the map in pixels
 myOrigin = param.origin; 
 
% Log-odd parameters 
lo_occ = param.lo_occ;
lo_free = param.lo_free; 
lo_max = param.lo_max;
lo_min = param.lo_min;

N = size(pose,2);

%Create figure window
figure('units','pixels','position',get(groot,'ScreenSize'),'Name','Grid Map','NumberTitle','off');

%Preallocation of all Poses
allPoses = [];

%Video capturing check
if (param.captureBool)
    fileName = ['record_', datestr(now,'yyyy-mm-dd_HH-MM-SS'), '.mp4'];
    videoObj = VideoWriter(fileName,'MPEG-4');
    videoObj.Quality=100;
    videoObj.FrameRate = 3;
    open(videoObj);
end

zg = zeros(35, 50);
for j = 1:N 
    %position of pose on grid
    gridPose = ceil(myResol*[pose(1,j);pose(2,j)]) + myOrigin;
    %Find occupied-measurement cells from LIDAR readings
    occCells = grids(ranges(:,j),scanAngles,pose(:,j),myOrigin,myResol);
    
    %https://cn.mathworks.com/matlabcentral/answers/73126-contour-plot-from-coordinates
%     xx = occCells(:,1);
%     yy = occCells(:,2);
%     [xi, yi] = meshgrid(xx(1:100), yy(1:100));
%     Z = griddata(xx(1:100), yy(1:100), height(:,j), xi, yi);
    
%     mp = 1;
%     for p=occCells(:,1)
%         for k=occCells(:,2)
%             Z(p,k) = height(mp,j);
%             mp = mp+1;
%         end
%     end

    % Find free-measurement cells
    for i = 1 : length(occCells)
        [freex, freey] = bresenham(gridPose(1),gridPose(2),occCells(i,1),occCells(i,2)); 
        free = sub2ind(size(myMap),freey,freex);
        % Update the log-odds
        myMap(occCells(i,2),occCells(i,1)) = myMap(occCells(i,2),occCells(i,1)) + lo_occ;
        myMap(free) = myMap(free) - lo_free;
    end

    % Saturate the log-odd values
    myMap(myMap>lo_max)= lo_max;
    myMap(myMap<lo_min)= lo_min;
    
    
    % Visualize the map as needed
    allPoses = [allPoses gridPose]; %current + previous poses
  
    colormap gray;
    %imagesc(myMap);
    %C = uint16(nma_rescale(myMap, 0, ));
    %CX = imcomplement(C);
    CX = max(myMap(:))-myMap+min(myMap(:));
    imagesc(CX);
    hold on;
   % axis equal;
    
    plot(allPoses(1,:),allPoses(2,:),'r.-','LineWidth',2.5); %Plot the poses up till this point
    %plot3(allPoses(1,:),allPoses(2,:),zeros(length(allPoses(1,:)), 1), 'r.-','LineWidth',2.5);
    lidar_global(:,1) =  (ranges(:,j).*cos(scanAngles + pose(3,j)) + pose(1,j))*myResol + myOrigin(1);
    lidar_global(:,2) = (-ranges(:,j).*sin(scanAngles + pose(3,j)) + pose(2,j))*myResol + myOrigin(2);
    %plot(lidar_global(:,1), lidar_global(:,2), 'g.'); 
   % contour3(xi, yi, Z, 'g-');
    
    zg = writeText(zg, occCells, height(:,j));
    hold off;
    pause(0.1);
    %saveas(gcf, fullfile('E:\programming\dataSet\frames',strcat(num2str(j),'.png')));
   % drawnow;
    %Capture Video
    if (param.captureBool)
        writeVideo(videoObj, getframe(gcf));
    end
    
end

if (param.captureBool)
    close(videoObj);
end

end

