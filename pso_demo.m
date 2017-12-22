% Ref: Xin-She Yang. 2014.  Nature-Inspired Optimization Algorithms,
% Elsevier, Waltham, MA.
% Phu Dang
% CSCI 5454- Final project

function[best] = pso_demo(functionName,strfunc,numparticles,iter,x1min,x1max,x2min,x2max)
% Create function
f = vectorize(inline(strfunc));
% Create range for x1, x2
range = [x1min x1max x2min x2max];
beta = 0.5;
gamma = 0.7;
% Visualization code
grid = 100;
dx1 = (range(2)-range(1))/grid;
dx2 = (range(4)-range(3))/grid;
x1grid = range(1):dx1:range(2);
x2grid = range(3):dx2:range(4);
[x1,x2]=meshgrid(x1grid,x2grid);
y = f(x1,x2);
figure(1);
surfc(x1,x2,y);
best = zeros(iter,3);
% PSO Algorithm
% Init locations
x1range = range(2)-range(1);
x2range = range(4)-range(3);
x1n = rand(1,numparticles)*x1range+range(1);
x2n = rand(1,numparticles)*x2range+range(3);
figure(2);

writeObj = VideoWriter(functionName);
writeObj.FrameRate = 2;
open(writeObj);

% Iteration
for i = 1:iter
    contour(x1,x2,y,15);hold on;
    % Find the best location
    yn = f(x1n,x2n);
    yn_min = min(yn);
    x1_0 = min(x1n(yn==yn_min));
    x2_0 = min(x2n(yn==yn_min));
    y_0 = min(yn(yn==yn_min));
    % Plot
    plot(x1n,x2n,'.',x1_0,x2_0,'*');axis(range);
    
    frame = getframe(gcf);
    writeVideo(writeObj,frame);
    
    pause(1.0);
    alpha = gamma.^i;
    % Move particles
    nn = size(x2n,2);
    x1n = x1n.*(1-beta)+x1_0.*beta+alpha.*(rand(1,nn)-0.5);
    x2n = x2n.*(1-beta)+x2_0.*beta+alpha.*(rand(1,nn)-0.5);
    nn = length(x2n);
    for j = 1:nn
        if x1n(j)<=range(1), x1n(j)=range(1); end
        if x1n(j)>=range(2), x1n(j)=range(2); end
        if x2n(j)<=range(3), x2n(j)=range(3); end
        if x2n(j)>=range(4), x2n(j)=range(4); end
    end
    drawnow;
    hold off;
    best(i,1)=x1_0;
    best(i,2)=x2_0;
    best(i,3)=y_0;
end
close(writeObj);
        
    
