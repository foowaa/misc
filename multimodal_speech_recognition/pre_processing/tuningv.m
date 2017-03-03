clc 
clear all
cd F:\database\AVD\pic\pic


saved = zeros(9,10,6);
%%
for n1=1:6
    for n2=0:9
        for n3=1:9
            target=strcat('S',num2str(n1),'_',num2str(n2),'_0',num2str(n3),'*.bmp');
            files = dir(target);
            saved(n3,n2+1,n1)=length(files);
        end
    end
end



%%
for n1=1:6
    for n2=0:9
        for n3=1:9
            for i=1:saved(n3,n2+1,n1)
                if i>40
                target = strcat('S',num2str(n1),'_',num2str(n2),'_0',num2str(n3),'_',num2str(i),'.bmp');
                delete(target);
         %       else
                end
            end
        end
    end
end