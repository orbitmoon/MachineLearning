function M = meanpool(fm,W,B)

m = size(fm,1);
M = zeros(m/2,m/2);

for i = 1:(m/2)
    for j = 1:(m/2)
        M(i,j) = W*sum((fm(2*i-1:2*i,2*j-1:2*j))(:))/4 + B;
    end;
end;