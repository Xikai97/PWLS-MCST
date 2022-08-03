function cm = gammaCorrectColormap(cm,gamma)
% Syntax: cm = gammaCorrectColormap(cm,gamma);

% Parse inputs
n = size(cm,1);

% Gamma correct color
x = linspace(0,1,n);
tmp = nan(size(cm));
for i = 1:3
    tmp(:,i) = pchip(x,cm(:,i),x.^gamma);
end
cm = tmp;
