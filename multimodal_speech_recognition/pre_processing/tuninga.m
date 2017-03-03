savedm = saved1;
aud = zeros(21600,200);
for i=1:540
    st = savedm{i,1};
    for j=1:40
        x = 40*(i-1)+j;
        aud(x,:) = cat(2,st(j,:),st(j+1,:),st(j+2,:),st(j+3,:));
    end
end
        